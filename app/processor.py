"""Orchestrates the full video-analysis pipeline from ingest through interaction detection."""

# ──────────────────────────────────────────────────────────────────────────────
# Responsibility of this module
#
# processor.py is the single entry point for video analysis. It calls the four
# pipeline stages in order, applies cross-cutting filters (track length), builds
# the authoritative output dict, and returns it ready to be persisted by the
# database layer.
#
# Nothing in this file touches HTTP or SQLite directly — those are concerns of
# main.py and database.py respectively. Keeping that boundary clean means you
# could swap the API layer for a CLI script or a Celery worker without changing
# any pipeline logic.
# ──────────────────────────────────────────────────────────────────────────────

import logging
from pathlib import Path
from typing import Any, Dict, List

from app.models import (
    DetectedObject,
    Interaction,
    KeyFrame,
    MotionInterval,
    TaskResult,
    VideoMetadata,
)
from app.pipeline.detection import Detector
from app.pipeline.interaction import detect_interactions
from app.pipeline.keyframes import extract_keyframes
from app.pipeline.motion import classify_motion

logger = logging.getLogger(__name__)

# Resolve the outputs directory once at import time. JPEGs extracted by the
# keyframe stage are written into a per-task subdirectory under this root.
# Pattern: data/outputs/{task_id}/obj{id}_frame{n:06d}_{reason}.jpg
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUTS_DIR  = _PROJECT_ROOT / "data" / "outputs"


def process_video(
    task_id: str,
    video_path: str,
    frame_stride: int = 1,
    min_track_observations: int = 8,
) -> Dict[str, Any]:
    """Run the full analysis pipeline on one video and return a serialised TaskResult.

    Stages
    ------
    1. Detection + tracking   — YOLO + ByteTrack, produces raw tracks.
    2. Track-length filter    — drops transient/noise tracks.
    3. Motion classification  — per-track moving/stationary segments.
    4. Interaction detection  — sustained person-object proximity episodes.
    5. Schema assembly        — builds and validates a TaskResult Pydantic model.

    Parameters
    ----------
    task_id:
        Used for log correlation only. Every log line from this call is tagged
        with the task_id so you can grep a single job's trace out of mixed output.

    video_path:
        Absolute path to the uploaded video file saved in data/uploads/.

    frame_stride:
        Every N-th frame is recorded as an observation. Passed through to
        Detector.detect_and_track. The tracker still processes every frame;
        stride only controls how many positions we store. Higher stride = faster
        but coarser motion history. Default 1 = every frame.

    min_track_observations:
        Tracks with fewer than this many observations are dropped before any
        further processing.

        Why filter here rather than inside the detector?

        The detector produces all confirmed ByteTrack tracks, including short-lived
        ones from:
          - Partial occlusion (an arm briefly crosses the object)
          - Reflections or shadows that briefly trigger a detection
          - People walking through the edge of frame (1–3 second tracks)
          - Ghost tracks: ByteTrack confirmed a track then immediately lost it

        These would appear as objects in the final output with 0 interactions and
        a 1-segment motion history, making the result noisy and misleading.
        Filtering at this stage — after detection, before any downstream analysis —
        keeps the authoritative output clean without affecting tracker quality.

        Default 8 observations: at frame_stride=1 and 30 fps, this requires the
        object to be visible for at least 8 frames (~0.27 s) — enough to exclude
        single-frame spurious detections while keeping all real interactions which
        typically last seconds.

    Returns
    -------
    A plain Python dict produced by TaskResult.model_dump(by_alias=True), ready
    to be passed to database.save_task_result(). It matches the API schema exactly,
    including "class" (not "object_class") as the JSON key for object labels.
    """

    logger.info("[%s] Pipeline started. video=%s stride=%d min_obs=%d",
                task_id, video_path, frame_stride, min_track_observations)

    # ── Stage 1: Detection + tracking ────────────────────────────────────────
    #
    # Why create a new Detector per call rather than caching it?
    #
    # A module-level singleton would be more efficient (weights load once at
    # startup rather than once per video), but it introduces thread-safety risk:
    # if two tasks run concurrently and both call model.track() on the same YOLO
    # instance, the ByteTrack internal state (persist=True) can corrupt between
    # calls. Creating a fresh Detector per call avoids any shared mutable state.
    #
    # Production answer: use a worker process pool (Celery + Redis) where each
    # worker owns exactly one Detector instance and processes tasks serially.
    detector = Detector()
    metadata_dict, raw_tracks = detector.detect_and_track(
        video_path, frame_stride=frame_stride
    )
    logger.info("[%s] Detection done: %d raw tracks", task_id, len(raw_tracks))

    # ── Stage 2: Track-length filter ─────────────────────────────────────────

    tracks: Dict[int, Any] = {
        tid: record
        for tid, record in raw_tracks.items()
        if len(record["observations"]) >= min_track_observations
    }

    n_dropped = len(raw_tracks) - len(tracks)
    logger.info("[%s] After length filter: %d tracks kept, %d dropped (< %d obs)",
                task_id, len(tracks), n_dropped, min_track_observations)

    # ── Stage 3: Person / object split ────────────────────────────────────────
    #
    # detect_interactions needs the two groups separately so it can evaluate
    # every person-object pair. We compute the split once and reuse below.

    person_tracks: Dict[int, Any] = {
        tid: r for tid, r in tracks.items() if r["is_person"]
    }
    object_tracks: Dict[int, Any] = {
        tid: r for tid, r in tracks.items() if not r["is_person"]
    }
    logger.info("[%s] %d person(s), %d object(s)",
                task_id, len(person_tracks), len(object_tracks))

    # ── Stage 4: Interaction detection ───────────────────────────────────────
    #
    # Why run interaction detection before motion classification?
    #
    # No particular reason for this order — the two stages are independent.
    # We run interactions first because it is the costlier O(P × O × F) loop
    # (P persons × O objects × F shared frames) and we want to log its result
    # before the per-track O(T × F) motion passes.
    #
    # Why detect_interactions only receives surviving tracks?
    #
    # A filtered-out ghost track (< min_track_observations) should not generate
    # interactions. Passing raw_tracks instead of tracks would let a 2-observation
    # phantom person appear in the interactions list, which is exactly the noise
    # we're trying to suppress.

    interactions_by_object: Dict[int, List[Dict[str, Any]]] = {}
    if person_tracks and object_tracks:
        interactions_by_object = detect_interactions(person_tracks, object_tracks)
        n_total = sum(len(v) for v in interactions_by_object.values())
        logger.info("[%s] %d interaction(s) across %d object(s)",
                    task_id, n_total, len(interactions_by_object))
    else:
        logger.info("[%s] Skipping interaction detection (no persons or no objects)",
                    task_id)

    # ── Stage 5: Assemble the output schema ───────────────────────────────────

    detected_objects: List[DetectedObject] = []

    # Iterate sorted by track_id so the output order is stable and predictable.
    # This matters for result diffing and regression testing.
    for track_id, record in sorted(tracks.items()):

        # ── Motion history ─────────────────────────────────────────────────
        motion_segments = classify_motion(record["observations"])
        motion_intervals = [
            MotionInterval(**seg)  # seg = {"frame_range": [...], "state": "..."}
            for seg in motion_segments
        ]

        # ── Interactions ───────────────────────────────────────────────────
        #
        # Why are interactions recorded on the OBJECT rather than the person?
        #
        # The schema models the question "what interacted with this object?" which
        # is the natural query in a lab monitoring context: "who touched the
        # centrifuge?" not "what did this person touch?".
        # The Interaction model stores interacted_by_person so you can still
        # reconstruct "everything this person touched" by querying across all
        # objectsDetected, but the primary index is the object.
        #
        # Why is the PERSON's interactions list empty here?
        #
        # detect_interactions returns {object_id: [...interactions...]}. A person's
        # track_id is never an object_id, so interactions_by_object.get(person_id)
        # always returns the default empty list. This is intentional: the person is
        # a detected entity (it has an object_id, class, motion_history) but the
        # schema doesn't ask "who did this person interact with?" — it asks "who
        # interacted with this object?". Including persons in objectsDetected with
        # empty interactions is the clean, uniform treatment.
        raw_interactions = interactions_by_object.get(track_id, [])
        interaction_models = [
            Interaction(**iv)  # iv = {"interacted_by_person": ..., "frame_start": ..., ...}
            for iv in raw_interactions
        ]

        # ── Why is the person included in objectsDetected at all? ─────────
        #
        # The schema treats every tracked entity uniformly: it has an object_id,
        # a class label, a motion history, and an interactions list. The person
        # just happens to have class="person" and empty interactions. This uniform
        # treatment means the client can iterate objectsDetected to build a full
        # timeline for everything in the scene — including who was moving and when
        # — without needing a separate "persons" endpoint.

        detected_objects.append(
            DetectedObject(
                object_id=track_id,
                # Use the Python field name (object_class), not the alias ("class").
                # populate_by_name=True on DetectedObject allows this. Pydantic maps
                # object_class -> alias "class" when serialising with by_alias=True.
                object_class=record["class_label"],
                motion_history=motion_intervals,
                interactions=interaction_models,
            )
        )

    # ── Stage 6: Keyframe extraction (bonus feature — non-fatal) ─────────────
    #
    # Why is this wrapped in a try/except when the other stages are not?
    #
    # Keyframe extraction is an enhancement, not a core deliverable. The motion
    # history and interaction data computed above are the API contract; the JPEG
    # stills are supporting evidence for human review. If extraction fails — due
    # to a corrupt video segment, a full disk, an OpenCV codec edge case, or any
    # bug in the keyframe logic — that failure must not set the task to "failed"
    # and discard the correctly-computed pipeline results. The catch here converts
    # a fatal crash into a logged warning: the caller gets a complete TaskResult
    # with an empty keyFrames list rather than no result at all.
    #
    # General principle: non-essential features should degrade gracefully so they
    # can never jeopardise the output that the core pipeline has already earned.

    keyframe_models: List[KeyFrame] = []
    try:
        task_output_dir = _OUTPUTS_DIR / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)

        raw_keyframes = extract_keyframes(
            video_path=video_path,
            objects_detected=detected_objects,
            interactions_by_object=interactions_by_object,
            output_dir=task_output_dir,
            task_id=task_id,
            fps=metadata_dict.get("fps", 0.0),
        )
        keyframe_models = [KeyFrame(**kf) for kf in raw_keyframes]
        logger.info("[%s] Keyframe extraction: %d still(s) saved to %s",
                    task_id, len(keyframe_models), task_output_dir)

    except Exception as exc:
        logger.warning(
            "[%s] Keyframe extraction failed (non-fatal, continuing without stills): %s",
            task_id, exc,
        )
        keyframe_models = []

    # ── Stage 7: Build and validate the top-level result ─────────────────────
    #
    # Constructing the Pydantic model here (rather than just building a raw dict)
    # provides schema validation at write time: if any field is wrong type or missing,
    # Pydantic raises a ValidationError before we ever touch the database. This
    # means a bug in the pipeline produces a clear Python exception rather than
    # silently writing malformed JSON to tasks.db.

    result = TaskResult(
        videoMetadata=VideoMetadata(**metadata_dict),
        objectsDetected=detected_objects,
        keyFrames=keyframe_models if keyframe_models else None,
    )

    logger.info(
        "[%s] Assembly complete: %d objects, %d total interactions",
        task_id,
        len(detected_objects),
        sum(len(obj.interactions) for obj in detected_objects),
    )

    # model_dump(by_alias=True) is critical: it serialises "object_class" -> "class"
    # in the JSON output. Without by_alias=True the client would receive the Python
    # field name "object_class" and the API contract would be violated.
    # This dict is what gets stored as result_json TEXT in the tasks table.
    return result.model_dump(by_alias=True)
