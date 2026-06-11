"""YOLO-based object and person detection — runs inference on individual video frames."""

# ──────────────────────────────────────────────────────────────────────────────
# Why YOLOv8 + ultralytics?
#
# YOLO ("You Only Look Once") is a family of single-pass object detectors.
# Unlike two-stage detectors (e.g. Faster R-CNN) that first propose regions then
# classify them, YOLO divides the image into a grid and predicts boxes + class
# probabilities in one forward pass. That makes it fast enough for real-time
# video on commodity hardware.
#
# The `ultralytics` library (made by the creators of YOLOv8) wraps the model in
# a clean Python API and handles weights download, ONNX export, and — crucially
# for us — built-in multi-object tracking via model.track().
# ──────────────────────────────────────────────────────────────────────────────

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2

# Ultralytics is imported at module level so an ImportError surfaces at startup,
# not silently at the first request.
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Type aliases — descriptive names on plain Python types.
# Zero runtime cost; they exist to make function signatures self-documenting
# and to give type-checkers like mypy something to reason about.
# ──────────────────────────────────────────────────────────────────────────────

# A bounding box in xyxy format: [x_top_left, y_top_left, x_bottom_right, y_bottom_right].
# All coordinates are pixel values relative to the original frame size.
# "xyxy" contrasts with "xywh" (centre-x, centre-y, width, height) — xyxy is
# easier to work with for intersection calculations downstream.
BBox = List[float]

# One sighting of a tracked object: the frame it appeared in + its bounding box.
Observation = Tuple[int, BBox]

# The full history for one tracked entity.
# Keys:
#   class_label    — authoritative label used by all downstream modules.
#                    Either a trusted COCO name (e.g. "person") or the generic
#                    string "object" for anything the model cannot reliably name.
#   raw_coco_label — the raw YOLO prediction before our honesty filter; kept for
#                    debugging and transparency, not used in any pipeline logic.
#   is_person      — pre-computed bool so downstream modules don't re-check the string.
#   observations   — time-ordered list of (frame_index, bbox) across the video.
TrackRecord = Dict[str, Any]

# The default set of COCO classes we trust the pretrained model to label correctly.
# "person" is the one class that is reliably detected by COCO-trained YOLO even
# in novel environments — human body shape is visually invariant enough that the
# model generalises well. Lab equipment (microscopes, centrifuges, pipettes) was
# never in the training distribution, so those predictions are not trustworthy.
DEFAULT_TRUSTED_CLASSES: Set[str] = {"person"}


class Detector:
    """Wraps a YOLOv8 model and exposes a single high-level method: detect_and_track.

    Keeping the model loading in __init__ means the heavyweight YOLO weights are
    loaded once when the service starts (or when the Detector is first instantiated),
    not on every video request. That avoids the multi-second cold-start penalty
    per task.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        trusted_classes: Optional[Set[str]] = None,
    ) -> None:
        """Load the YOLOv8 model and configure the OOD-handling policy.

        Parameters
        ----------
        model_name:
            Which YOLOv8 variant to use. "n" = nano (fastest, least accurate),
            "s/m/l/x" trade accuracy for speed. On first run, ultralytics
            auto-downloads the weights from GitHub to ~/.cache/ultralytics/.
            In production you'd pre-bake the weights into the Docker image so
            there's no network call at runtime.

            Why yolov8n specifically?
            For an interaction-detection service, we care more about reliably
            spotting *presence* (is something there?) than pixel-perfect
            segmentation. Nano runs at ~100 fps on a mid-range GPU and still
            achieves ~37 mAP on COCO, which covers all 80 standard classes
            including "person".

        confidence_threshold:
            Detections below this score are discarded before they ever reach
            ByteTrack. Default 0.5 is a reasonable starting point.

            Why does a confidence threshold help with out-of-distribution (OOD)
            objects?

            A COCO-pretrained YOLO has never seen lab equipment. When it
            encounters an oscilloscope it does two things: (a) it still fires a
            bounding box because *something* is there — YOLO is sensitive to
            "foreground objects" as a concept — but (b) the *classification*
            softmax spreads its probability mass across visually similar COCO
            classes ("laptop", "cell phone", "remote") and the resulting
            confidence score is noticeably lower than for in-distribution objects.

            Empirically, in-distribution objects (people, chairs, bottles)
            typically score >0.7. OOD objects tend to cluster below 0.5.
            Raising the threshold filters out many hallucinated false positives
            before they pollute the tracker's ID pool.

            This is not perfect — a high-contrast beaker might still score 0.6
            as a "cup" — which is why we also apply the trusted_classes filter
            at the labeling stage.

        trusted_classes:
            The subset of COCO labels we trust the pretrained model to emit
            correctly for this deployment. Every detected COCO class *not* in
            this set will be relabeled as the generic string "object".

            Why only trust "person" by default?

            "Person" is one of the most heavily represented classes in COCO
            (~64,000 annotated instances vs. ~1,500 for many equipment-adjacent
            classes like "laptop"). The model has seen humans in every conceivable
            context and generalises extremely well. Lab equipment, on the other
            hand, was never in the training distribution.

            The honest thing to do is say "I can see there's *something* here
            and I can track it across frames, but I won't pretend to know it's
            a 'car'" — which is what relabeling to "object" achieves.

            In a production system you would fine-tune YOLO on a dataset that
            includes your specific equipment classes, then expand trusted_classes
            to include them. That change lives entirely in this __init__ — the
            motion, interaction, and processor modules are completely unaffected
            because they operate on class_label, which remains either a trusted
            string or "object" either way.
        """
        logger.info("Loading YOLO model: %s", model_name)
        self.model = YOLO(model_name)
        logger.info("Model loaded. Classes available: %d", len(self.model.names))

        self.confidence_threshold = confidence_threshold

        # Use the caller's set if provided, otherwise fall back to the module-level
        # default. We copy the set so external mutation after construction has no
        # effect — a Detector's policy should be immutable after __init__.
        self.trusted_classes: Set[str] = (
            set(trusted_classes) if trusted_classes is not None else set(DEFAULT_TRUSTED_CLASSES)
        )

        logger.info(
            "OOD policy — confidence threshold: %.2f, trusted classes: %s",
            self.confidence_threshold,
            sorted(self.trusted_classes),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Main public method
    # ──────────────────────────────────────────────────────────────────────────

    def detect_and_track(
        self,
        video_path: str,
        frame_stride: int = 1,
    ) -> Tuple[Dict[str, Any], Dict[int, TrackRecord]]:
        """Process a video file and return per-object tracking data.

        Steps
        -----
        1. Open the video with OpenCV to extract metadata (fps, resolution, etc.).
        2. Feed the entire video through YOLOv8 + ByteTrack to get per-frame
           detections with stable track IDs. Low-confidence detections are
           discarded by the model before the tracker sees them.
        3. For each confirmed track, apply the OOD labeling policy:
           trusted COCO class -> keep it; unknown class -> relabel as "object".
        4. Accumulate per-track observation histories, respecting frame_stride.
        5. Return (metadata_dict, tracks_dict).

        Parameters
        ----------
        video_path:
            Absolute or relative path to the video file.
        frame_stride:
            How often to *record* an observation.
            stride=1  -> record every frame (max accuracy, max data volume)
            stride=5  -> record every 5th frame (5x less data, slight position drift)

            IMPORTANT: The tracker still *processes* every frame regardless of
            stride. Skipping frames fed into ByteTrack would break its Kalman-
            filter motion model — tracks would flicker and IDs would be reassigned
            incorrectly. stride only controls how often we *store* the bounding
            box in the observation list, not how often the model runs.

        Returns
        -------
        metadata : dict
            {"duration": float, "frame_count": int, "resolution": "WxH", "fps": float}
        tracks : dict[track_id -> TrackRecord]
            Keyed by integer track ID. Each value is a TrackRecord with
            "class_label", "raw_coco_label", "is_person", and "observations".
        """
        video_path = str(video_path)  # accept Path objects transparently

        # Step 1: Extract video metadata with OpenCV.
        metadata = self._read_video_metadata(video_path)

        # Steps 2-4: Track objects, apply OOD labeling, accumulate histories.
        tracks = self._run_tracking(video_path, frame_stride)

        n_people = sum(1 for r in tracks.values() if r["is_person"])
        n_objects = len(tracks) - n_people
        n_relabeled = sum(
            1 for r in tracks.values()
            if not r["is_person"] and r["raw_coco_label"] != "object"
        )
        logger.info(
            "Finished %s — %d tracks (%d people, %d objects, %d relabeled from COCO guess)",
            Path(video_path).name,
            len(tracks),
            n_people,
            n_objects,
            n_relabeled,
        )
        return metadata, tracks

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _read_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Open the video with OpenCV and read its container-level properties.

        Why OpenCV instead of FFprobe?
        OpenCV is already in our dependency tree (ultralytics pulls it in).
        cv2.VideoCapture gives us the same properties with zero extra deps.
        It reads the container header — no frames are decoded here.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video file: {video_path}")

        try:
            # CAP_PROP_* are integer constants that act as "keys" for VideoCapture.get().
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            # Always release — a leaked VideoCapture holds a file handle.
            cap.release()

        # Guard against malformed containers that report fps=0.
        duration = round(frame_count / fps, 3) if fps > 0 else 0.0

        return {
            "duration": duration,
            "frame_count": frame_count,
            # "WxH" format, e.g. "1920x1080". The interaction module uses this
            # to normalise bounding-box coordinates to relative [0,1] space.
            "resolution": f"{width}x{height}",
            "fps": fps,
        }

    def _resolve_label(self, raw_coco_label: str) -> str:
        """Apply the OOD honesty policy to a single raw COCO class prediction.

        This is the entire labeling decision in one place — a deliberate single
        point of responsibility so that changing the policy (e.g. expanding
        trusted_classes after fine-tuning) means editing exactly one method.

        Why keep pipeline code decoupled from this decision?
        The motion module cares about bounding-box displacement, not label names.
        The interaction module cares whether something is a "person", which is
        captured by is_person. Neither module inspects class_label beyond that.
        Relabeling OOD objects as "object" is therefore a zero-impact change for
        all downstream code — it only affects what we *report*, not what we track.
        """
        if raw_coco_label in self.trusted_classes:
            return raw_coco_label
        # The model's COCO guess is unreliable for this deployment. Return a
        # non-committal label rather than confidently emitting a wrong category.
        return "object"

    def _run_tracking(
        self, video_path: str, frame_stride: int
    ) -> Dict[int, TrackRecord]:
        """Feed the video through YOLOv8 + ByteTrack and return the track dict.

        This is the heart of the module. Each argument to model.track() is
        deliberately chosen and worth understanding for an interview.
        """

        # ── Why model.track() instead of model.predict() + rolling our own tracker?
        #
        # model.predict() returns one independent detection result per frame — it
        # has no memory of previous frames. To know that the "chair" in frame 10
        # is the same chair as in frame 20, you need a tracker that links detections
        # across time. You could hand-roll this with IoU matching, but ByteTrack is
        # a well-validated implementation with better handling of occlusion and
        # re-identification. Using model.track() gives us all of that for free.

        # ── Why persist=True?
        #
        # ByteTrack maintains internal state: a Kalman filter per track that
        # predicts where each object will be in the next frame, and a "lost" buffer
        # for objects that temporarily disappear (e.g. walk behind a pillar).
        # persist=True tells the tracker to carry that state across generator
        # iterations. Without it, each next() call would start a fresh tracker
        # and reassign IDs from 1, defeating the purpose entirely.

        # ── Why ByteTrack instead of DeepSORT?
        #
        # DeepSORT uses a learned re-identification embedding (a small CNN) to match
        # lost tracks. That extra inference pass costs ~30% throughput.
        # ByteTrack achieves similar re-id accuracy using only IoU + Kalman motion —
        # no extra network — by cleverly keeping "low confidence" detections in a
        # secondary matching pool. For a fixed-camera lab scene ByteTrack is faster
        # and equally accurate.

        # ── Why stream=True?
        #
        # Without stream=True, model.track() decodes the *entire* video into RAM,
        # runs inference on every frame, and returns a list of all Results objects
        # before your code sees any of them. A 10-minute 1080p video is ~18 GB of
        # raw frames — that will OOM a typical server.
        # stream=True returns a generator: it yields one Results object per frame,
        # processes the next frame only when you call next(), and keeps only one
        # decoded frame in GPU memory at a time. This is the mandatory pattern for
        # any video longer than a few seconds.

        # ── Why conf=self.confidence_threshold?
        #
        # Passing conf here applies the threshold *inside* the YOLO inference pass,
        # before ByteTrack even sees the detection. This is better than filtering
        # boxes after the fact because:
        #   1. Low-confidence boxes are never fed to the Kalman filter, so the
        #      tracker's internal state stays clean.
        #   2. Tracks that only ever have low-confidence sightings are never
        #      promoted to confirmed status — the noise never accumulates.
        # The net effect: OOD hallucinations that score below the threshold are
        # eliminated entirely, not just ignored post-hoc.

        results_gen = self.model.track(
            source=video_path,
            persist=True,
            tracker="bytetrack.yaml",         # shipped with ultralytics, no extra file needed
            stream=True,                       # generator — O(1) memory regardless of length
            conf=self.confidence_threshold,    # drop low-confidence detections at inference time
            verbose=False,                     # suppress per-frame progress bar on stdout
        )

        tracks: Dict[int, TrackRecord] = {}

        for frame_idx, result in enumerate(results_gen):
            boxes = result.boxes

            # ── Why might boxes.id be None?
            #
            # ByteTrack uses a two-stage confirmation policy: a newly appearing
            # object must be detected in consecutive frames before it's promoted to
            # a "confirmed" track and assigned a stable integer ID. On the very
            # first frame, or when the scene has only fleeting detections,
            # boxes.id can be None even if boxes.xyxy has entries.
            # We skip those unconfirmed detections — they'd produce track ID 0 or
            # crash on .tolist().
            if boxes is None or boxes.id is None:
                continue

            # ── Convert PyTorch tensors to plain Python types ─────────────────
            #
            # ultralytics returns everything as torch.Tensor on whatever device the
            # model is running (CPU or CUDA). .tolist() moves the data to CPU and
            # converts it to a nested Python list — after this point there is no
            # PyTorch dependency in the rest of the function.
            track_ids: List[int] = [int(t) for t in boxes.id.tolist()]
            class_ids: List[int] = [int(c) for c in boxes.cls.tolist()]

            # ── xyxy bounding box format explained ────────────────────────────
            #
            # Each bbox is [x1, y1, x2, y2] where:
            #   (x1, y1) = top-left corner  (pixel coordinates, origin at top-left)
            #   (x2, y2) = bottom-right corner
            # A box covering the full 1920x1080 frame would be [0, 0, 1920, 1080].
            #
            # Why xyxy and not xywh (centre + size)?
            # For intersection-over-union (IoU) and proximity tests in the
            # interaction module, xyxy is more direct: overlap_rect = (max(x1s),
            # max(y1s), min(x2s), min(y2s)) — one line of arithmetic. xywh needs
            # an extra conversion step first.
            bboxes: List[BBox] = boxes.xyxy.tolist()

            # Decide whether to *record* this frame's observation.
            # Computed once outside the inner loop — same for every object in this frame.
            should_record = (frame_idx % frame_stride == 0)

            for track_id, class_id, bbox in zip(track_ids, class_ids, bboxes):

                # ── OOD labeling policy ───────────────────────────────────────
                #
                # raw_coco_label is what YOLO actually predicted. class_label is
                # what we authoritatively report downstream. The two are identical
                # for trusted classes and diverge for everything else.
                #
                # This split matters for transparency: the API response can include
                # raw_coco_label as a debug field so a human reviewer can see "YOLO
                # thought this was a laptop, but we relabeled it as object" — rather
                # than silently swallowing the incorrect prediction.
                raw_coco_label: str = self.model.names[class_id]
                class_label: str = self._resolve_label(raw_coco_label)
                is_person: bool = class_label == "person"

                # ── First sighting of this track ID -> create its record ──────
                #
                # We always create the record on first sight even if we're not
                # going to record the observation (frame_stride > 1). That way the
                # track exists in the dict from the moment it's confirmed, and
                # downstream code can iterate all tracks without gap concerns.
                #
                # Note: we use the label from the *first* confirmed sighting. In
                # theory ByteTrack could match a person detection to a previous
                # "object" track if the model briefly misclassifies, but in practice
                # confirmed person tracks are stable enough that this doesn't occur.
                # A production system could use majority-vote across all sightings.
                if track_id not in tracks:
                    tracks[track_id] = {
                        "class_label": class_label,
                        "raw_coco_label": raw_coco_label,
                        "is_person": is_person,
                        "observations": [],
                    }

                if should_record:
                    # Appending a tuple keeps the observations list lean.
                    # (frame_idx, bbox) is the minimal data needed by both the
                    # motion module (position over time) and the interaction module
                    # (position at a specific moment).
                    tracks[track_id]["observations"].append((frame_idx, bbox))

        return tracks
