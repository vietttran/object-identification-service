"""Extracts representative JPEG stills from a video at notable moments in tracked objects' histories."""

# ──────────────────────────────────────────────────────────────────────────────
# Why extract keyframes at all?
#
# The core pipeline output — motion intervals, interaction episodes — is expressed
# purely as frame-index numbers. That data is precise and machine-readable, but
# it is hard to visually verify without scrubbing through the original video:
#
#   "Was frame 45 really when the person started moving?"
#   "What does the interaction look like at its midpoint?"
#
# A small set of JPEG stills that answer these questions directly gives a human
# reviewer — or an interview panel — immediate visual evidence to validate (or
# challenge) the algorithm's output. That is the purpose of this module.
#
# Two types of "notable moment" are extracted:
#
#   1. Stationary → Moving transition
#      The first frame of each new "moving" segment is the exact frame where the
#      displacement threshold was crossed. Saving it lets a reviewer answer:
#      "Does this still actually look like the start of motion?" If so, the
#      threshold is calibrated correctly. If the person is clearly stationary,
#      something is wrong with jitter handling or the threshold value.
#
#   2. Interaction peak (midpoint of interaction interval)
#      Our interaction signal is binary: a frame is either "in contact" or "not
#      in contact". We produce no within-interval confidence score. A pose
#      estimator (MediaPipe Hands, OpenPose) would give a continuous wrist-contact
#      probability from which we could select the maximum — but that model is not
#      in this pipeline.
#
#      The temporal midpoint of each interaction interval is our proxy for peak
#      engagement. The reasoning:
#        - The first few frames of the interval are the approach phase; the person
#          may still be leaning in and not fully engaged.
#        - The last few frames are the withdrawal phase; the person has started
#          to pull away.
#        - The midpoint sits past the approach and before the withdrawal — the
#          most plausible moment of full engagement.
#      This is documented explicitly as an approximation. The reason field in the
#      output is named "interaction_peak" (intent) rather than "interaction_midpoint"
#      (implementation detail) so that the API contract can be honoured even if
#      the selection logic is improved in a future version.
# ──────────────────────────────────────────────────────────────────────────────

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2  # type: ignore  # OpenCV Python stubs are not in the standard index

logger = logging.getLogger(__name__)


def extract_keyframes(
    video_path: str,
    objects_detected: List[Any],
    interactions_by_object: Dict[int, List[Dict[str, Any]]],
    output_dir: Path,
    task_id: str,
    fps: float = 0.0,
) -> List[Dict[str, Any]]:
    """Identify notable frames in the pipeline output, seek to them, and save JPEGs.

    Parameters
    ----------
    video_path:
        Absolute path to the uploaded video file. Passed directly to
        cv2.VideoCapture; any container supported by the system's OpenCV build
        (MP4, AVI, MOV, MKV, …) is accepted.

    objects_detected:
        The assembled List[DetectedObject] Pydantic models from processor.py.
        Each object exposes:
          .object_id: int
          .motion_history: List[MotionInterval]  (each has .state and .frame_range)
        Passed as List[Any] to avoid importing models.py here (no circular dep,
        just simpler — duck typing is fine for attribute access on Pydantic models).

    interactions_by_object:
        The raw dict {object_id: [{"frame_start": int, "frame_end": int, ...}]}
        produced by detect_interactions(). We only read frame_start/frame_end.

    output_dir:
        Absolute path to the per-task directory where JPEGs are written
        (data/outputs/{task_id}/). The directory must exist before calling this
        function. Caller is responsible for mkdir.

    task_id:
        Used to build the relative image_path stored in each returned keyframe
        dict, per the KeyFrame schema: "Relative path inside data/outputs/".
        Format: "{task_id}/{filename}".

    fps:
        Frames per second of the source video. Used for debug logging only
        (to convert frame numbers to human-readable timestamps). Does not affect
        seek logic, which always uses frame indices directly via CAP_PROP_POS_FRAMES.

    Returns
    -------
    List of dicts, each matching the KeyFrame Pydantic schema exactly:
        {"object_id": int, "frame_number": int, "reason": str, "image_path": str}

    Frames that cannot be read (seek past end, corrupt GOP, decode error) are
    silently skipped and omitted from the result. The caller (processor.py) wraps
    this entire function in a try/except so any unhandled exception here is also
    non-fatal.
    """

    # ── Step 1: Build the list of (object_id, frame_number, reason) targets ──

    targets: List[Tuple[int, int, str]] = []

    # ── 1a. Stationary → Moving transitions ─────────────────────────────────
    #
    # Scan each object's motion_history for a state change from "stationary"
    # to "moving". history[i].state == "stationary" and history[i+1].state ==
    # "moving" is the transition; the target frame is history[i+1].frame_range[0],
    # i.e. the FIRST frame of the "moving" segment.
    #
    # Why the first frame of the moving segment, not the last frame of the
    # stationary segment? They are often the same frame (the boundary is shared),
    # but the moving segment's start frame is the one that crossed the threshold,
    # so it is the most relevant still for verifying the decision boundary.
    #
    # Why not also capture Moving → Stationary? That transition is less ambiguous
    # (a person stopping is usually visually clear), and capturing both directions
    # would double the keyframe count for minimal diagnostic value. We focus on
    # the transition INTO motion because that is the harder decision to calibrate.
    for obj in objects_detected:
        history = obj.motion_history  # List[MotionInterval]
        for i in range(len(history) - 1):
            if history[i].state == "stationary" and history[i + 1].state == "moving":
                transition_frame = history[i + 1].frame_range[0]
                targets.append((obj.object_id, transition_frame, "stationary_to_moving"))
                logger.debug(
                    "[%s] Target: object %d stationary_to_moving at frame %d (%.2fs)",
                    task_id, obj.object_id, transition_frame,
                    transition_frame / fps if fps > 0 else 0,
                )

    # ── 1b. Interaction midpoints ────────────────────────────────────────────
    #
    # For each (person, object) interaction interval [frame_start, frame_end],
    # compute: midpoint = (frame_start + frame_end) // 2
    #
    # This is the "interaction_peak" proxy. See module-level docstring for the
    # full rationale. The key honest note: this midpoint is a heuristic, not the
    # result of maximising a continuous confidence signal. It is the best we can
    # do with a binary proximity-based interaction model without adding a second
    # model (pose estimator) to the pipeline.
    #
    # Integer floor division is used so the midpoint is always a valid frame index.
    for obj_id, interactions in interactions_by_object.items():
        for iv in interactions:
            midpoint = (iv["frame_start"] + iv["frame_end"]) // 2
            targets.append((obj_id, midpoint, "interaction_peak"))
            logger.debug(
                "[%s] Target: object %d interaction_peak at frame %d "
                "(midpoint of [%d, %d], %.2fs)",
                task_id, obj_id, midpoint,
                iv["frame_start"], iv["frame_end"],
                midpoint / fps if fps > 0 else 0,
            )

    if not targets:
        logger.info("[%s] No notable frames identified — keyframe extraction produces nothing.", task_id)
        return []

    # ── Step 2: Deduplicate ───────────────────────────────────────────────────
    #
    # The same (object_id, frame_number, reason) triple should never appear twice
    # in normal operation, but guard against it in case a motion transition and
    # an interaction midpoint happen to resolve to the same frame for the same
    # object with the same reason label.
    #
    # Note: (object_id=1, frame=45, reason="stationary_to_moving") and
    #       (object_id=1, frame=45, reason="interaction_peak") are NOT duplicates —
    # they represent distinct notable facts and will produce files with different names.
    seen: set = set()
    unique_targets: List[Tuple[int, int, str]] = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            unique_targets.append(t)

    logger.info("[%s] Keyframe targets: %d total, %d unique.",
                task_id, len(targets), len(unique_targets))

    # ── Step 3: Open the video and seek to each target frame ─────────────────
    #
    # cv2.VideoCapture opens the video via OpenCV's FFmpeg backend (on most
    # builds). It returns a capture object that supports random frame access via
    # the set() method.
    #
    # How frame seeking works (CAP_PROP_POS_FRAMES):
    #
    # cap.set(cv2.CAP_PROP_POS_FRAMES, n) moves the decode cursor to the n-th
    # frame (0-based index). For MP4/H.264 this involves finding the nearest
    # I-frame at or before frame n, then decoding forward to frame n if n is
    # between I-frames. Typical H.264 GOP sizes are 15–30 frames, so worst-case
    # seek cost is decoding ~1 second worth of frames — acceptable for a handful
    # of keyframes per video.
    #
    # Alternative: cap.set(cv2.CAP_PROP_POS_MSEC, ms) seeks by timestamp.
    # We use frame index instead because our entire pipeline works in frame space
    # and converting to milliseconds would introduce floating-point rounding.
    # Frame-index seeking is exact; timestamp seeking may land on the nearest
    # keyframe rather than the exact target.
    #
    # Optimisation note: for very long videos with many targets, sorting
    # unique_targets by frame_number and seeking linearly (using grab() to skip
    # frames rather than cap.set() for each) would be faster. For the expected
    # scale of this service (short lab clips, < 20 keyframes), random seek is
    # simpler and the performance difference is negligible.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("[%s] cv2.VideoCapture failed to open: %s", task_id, video_path)
        return []

    keyframe_results: List[Dict[str, Any]] = []

    try:
        for object_id, frame_number, reason in unique_targets:

            # Seek to the target frame. The C API expects a double; int works
            # in practice but the float cast silences potential C extension warnings.
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_number))

            # cap.read() = cap.grab() + cap.retrieve() in one call.
            # Returns (True, frame_array) on success, (False, None) on failure.
            # Failure modes: seek landed past end of file, the specific frame
            # belongs to a corrupt segment, or a codec error occurred mid-decode.
            ret, frame = cap.read()
            if not ret:
                logger.warning(
                    "[%s] Failed to read frame %d (object %d, %s) — skipping.",
                    task_id, frame_number, object_id, reason,
                )
                continue

            # Filename encodes all three key fields so the JPEG is self-describing.
            # Zero-padding to 6 digits (max 999,999 frames ≈ 9+ hours at 30 fps)
            # keeps files in correct lexicographic sort order.
            filename = f"obj{object_id}_frame{frame_number:06d}_{reason}.jpg"
            abs_path = output_dir / filename

            # cv2.imwrite returns True on success. It can return False (not raise)
            # if the path is not writable or the codec cannot encode the frame.
            success = cv2.imwrite(str(abs_path), frame)
            if not success:
                logger.warning(
                    "[%s] cv2.imwrite failed for %s — skipping.", task_id, abs_path
                )
                continue

            # image_path is relative to data/outputs/, per the KeyFrame schema.
            image_path = f"{task_id}/{filename}"

            keyframe_results.append({
                "object_id":    object_id,
                "frame_number": frame_number,
                "reason":       reason,
                "image_path":   image_path,
            })

    finally:
        # Always release the VideoCapture even if we exit via exception.
        # VideoCapture holds a file handle and a decoder context; leaking it
        # would exhaust OS file descriptors under repeated calls.
        cap.release()

    logger.info(
        "[%s] Keyframe extraction: %d/%d stills saved to %s",
        task_id, len(keyframe_results), len(unique_targets), output_dir,
    )
    return keyframe_results
