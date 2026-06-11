"""Classifies tracked objects as stationary or moving based on bounding-box displacement."""

# ──────────────────────────────────────────────────────────────────────────────
# Overview of the algorithm
#
# Input: the observation list for one tracked object — a sequence of
#   (frame_index, [x1, y1, x2, y2]) tuples produced by detection.py.
#
# Output: a list of MotionInterval-compatible dicts, each covering a contiguous
#   frame range with a single state label: "moving" or "stationary".
#
# The three-stage pipeline:
#   1. Measure  — compute per-interval center displacement, normalised per frame.
#   2. Segment  — group consecutive same-state intervals into contiguous blocks.
#   3. Smooth   — merge blocks shorter than min_segment_frames into their
#                 neighbour so that single-pair jitter doesn't create 1-step flickers.
# ──────────────────────────────────────────────────────────────────────────────

from math import sqrt
from typing import Any, Dict, List, Tuple

# Type aliases matching detection.py conventions.
BBox = List[float]              # [x1, y1, x2, y2]
Observation = Tuple[int, BBox]  # (frame_index, bbox)

# Internal segment representation used during processing.
# Each entry is a plain dict with:
#   "frame_range": [start_frame, end_frame]
#   "state":       "moving" | "stationary"
#   "_n":          number of observation-to-observation *intervals* that make up
#                  this segment — used only during smoothing, stripped before return.
_Segment = Dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────────
# Public function
# ──────────────────────────────────────────────────────────────────────────────

def classify_motion(
    observations: List[Observation],
    motion_threshold: float = 2.0,
    min_segment_frames: int = 3,
) -> List[Dict[str, Any]]:
    """Classify an object's motion history into contiguous moving/stationary segments.

    Parameters
    ----------
    observations:
        Time-ordered list of (frame_index, bbox) from detection.py.
        bbox is [x1, y1, x2, y2] in pixel coordinates.
        Observations can be non-consecutive (i.e. every N-th frame when
        frame_stride > 1 was used) — the algorithm handles gaps correctly.

    motion_threshold:
        Per-frame pixel displacement above which a frame-to-frame interval is
        labelled "moving". Default 2.0 pixels per frame.

        Why does this threshold need to exist at all?

        Even for a perfectly stationary object, a YOLO bounding box fluctuates
        slightly from frame to frame. The detector predicts slightly different
        corners on each pass because the CNN's receptive field is finite and the
        softmax output varies with sub-pixel changes in the input. This is called
        "detection jitter" or "box wobble". Empirically, a stationary box's
        centre drifts 0–2 pixels per frame. Setting the threshold above this
        noise floor prevents us from classifying motionless lab equipment as
        "moving" just because its YOLO box shifted by 1.5 pixels.

        If the threshold is too high, slow-moving people will be classified as
        stationary; too low, and even tripod-mounted cameras appear to move.
        2.0 px/frame is a reasonable default for 1080p footage; scale up for
        lower-resolution video where pixel distances are proportionally larger.

    min_segment_frames:
        Minimum number of observation-to-observation intervals a segment must
        span to be kept as its own state. Shorter segments are absorbed into
        their neighbour.

        IMPORTANT — this counts *intervals*, not raw video frames:
        An interval is one measurement step from observation[i] to observation[i+1].
        With frame_stride=5, each interval spans 5 frames, but it still counts
        as 1 interval. Using interval count instead of raw frame count makes
        the smoothing stride-independent: a 1-step jitter segment gets merged
        regardless of whether it covers 1 frame (stride=1) or 30 frames (stride=30).

        Why does smoothing matter?

        Suppose a lab worker holds their hand still for 0.5 s, then makes a
        tiny accidental twitch (one frame above threshold), then holds still again.
        Without smoothing this produces three segments: stationary → moving (1 interval)
        → stationary. With min_segment_frames=3, the 1-interval "moving" blip is
        absorbed into the neighbouring stationary segment and the result is one
        clean "stationary" block. This prevents state flicker that would otherwise
        generate spurious interaction events downstream.

    Returns
    -------
    A list of dicts, each matching the MotionInterval Pydantic schema:
        {"frame_range": [start_frame, end_frame], "state": "moving"|"stationary"}
    Adjacent entries never share the same state. The first entry's frame_range[0]
    equals the first observation's frame index; the last entry's frame_range[1]
    equals the last observation's frame index.
    """

    # ── Edge cases ────────────────────────────────────────────────────────────

    if not observations:
        # No observations at all: nothing to classify.
        return []

    if len(observations) == 1:
        # With only one sighting we cannot measure displacement. The object is
        # trivially stationary for the single frame we observed it.
        frame_idx, _ = observations[0]
        return [{"frame_range": [frame_idx, frame_idx], "state": "stationary"}]

    # ── Stage 1: Measure ──────────────────────────────────────────────────────

    raw_intervals = _measure_intervals(observations, motion_threshold)

    # ── Stage 2: Segment ──────────────────────────────────────────────────────

    segments = _merge_adjacent_states(raw_intervals)

    # ── Stage 3: Smooth ───────────────────────────────────────────────────────

    segments = _smooth_short_segments(segments, min_segment_frames)

    # Strip the internal "_n" count and return clean MotionInterval-compatible dicts.
    return [{"frame_range": seg["frame_range"], "state": seg["state"]}
            for seg in segments]


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bbox_center(bbox: BBox) -> Tuple[float, float]:
    """Return the (cx, cy) centre point of an xyxy bounding box.

    Why the centre and not a corner (e.g. the top-left)?

    Box *size* changes slightly from frame to frame due to detection jitter:
    the predicted width/height wobbles by a few pixels even for a stationary
    object. If we tracked a corner, that size wobble would directly add to the
    apparent displacement. The centre is the average of all four corner
    coordinates, so random corner wobble partially cancels out, giving a more
    stable anchor point.

    Example: if the box expands 4 pixels to the right but shrinks 4 pixels
    from the left, the top-left corner appears to move 4 px while the centre
    stays fixed.
    """
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return cx, cy


def _measure_intervals(
    observations: List[Observation],
    motion_threshold: float,
) -> List[_Segment]:
    """Compute one labelled interval for each consecutive pair of observations.

    Returns a list of raw single-interval segments (each with "_n": 1).
    Adjacent entries may have the same state; merging happens in the next stage.
    """
    intervals: List[_Segment] = []

    for i in range(len(observations) - 1):
        frame_i, bbox_i = observations[i]
        frame_j, bbox_j = observations[i + 1]

        cx_i, cy_i = _bbox_center(bbox_i)
        cx_j, cy_j = _bbox_center(bbox_j)

        # Euclidean distance between the two centre points in pixel space.
        displacement = sqrt((cx_j - cx_i) ** 2 + (cy_j - cy_i) ** 2)

        # ── Why divide by the frame gap? ──────────────────────────────────────
        #
        # With frame_stride=1 consecutive observations are 1 frame apart;
        # with frame_stride=5 they are 5 frames apart. An object travelling
        # at a constant speed covers 5× more distance in 5 frames.
        # Dividing by the gap normalises to "pixels per frame", so the same
        # motion_threshold applies regardless of the stride used during detection.
        #
        # Example: object moves 10 px over 5 frames → 2.0 px/frame.
        # With threshold=2.0 this is exactly on the boundary.
        # Without normalisation it would score 10 px (far above threshold) if stride=5
        # but 2 px per frame if stride=1 — an inconsistency that would make the
        # threshold meaningless for variable strides.
        frame_gap = frame_j - frame_i
        # Guard against duplicate frame indices (shouldn't happen, but be defensive).
        per_frame_disp = displacement / frame_gap if frame_gap > 0 else 0.0

        state = "moving" if per_frame_disp > motion_threshold else "stationary"

        intervals.append({
            "frame_range": [frame_i, frame_j],
            "state": state,
            "_n": 1,  # one interval = one measurement step
        })

    return intervals


def _merge_adjacent_states(intervals: List[_Segment]) -> List[_Segment]:
    """Collapse consecutive same-state intervals into single segments.

    This is a standard run-length encoding step. We walk the interval list
    once and either extend the current open segment or start a new one.
    The result has no two adjacent entries with the same state.
    """
    segments: List[_Segment] = []

    for iv in intervals:
        if segments and segments[-1]["state"] == iv["state"]:
            # Same state as the previous segment: extend its end frame and
            # increment the interval count.
            segments[-1]["frame_range"][1] = iv["frame_range"][1]
            segments[-1]["_n"] += 1
        else:
            # New state: start a fresh segment. Copy the frame_range list so
            # mutations below don't alias back to the intervals list.
            segments.append({
                "frame_range": list(iv["frame_range"]),
                "state": iv["state"],
                "_n": 1,
            })

    return segments


def _smooth_short_segments(
    segments: List[_Segment],
    min_segment_frames: int,
) -> List[_Segment]:
    """Iteratively absorb segments with fewer than min_segment_frames intervals.

    Algorithm
    ---------
    Repeat until no changes:
      1. Find the first segment whose "_n" count is below the threshold.
      2. Determine which neighbour to merge it into:
           - Leftmost segment  → merge right.
           - Rightmost segment → merge left.
           - Middle segment    → merge into the *shorter* neighbour
                                 (left wins on ties).
         The neighbour's state wins; the short segment's state is discarded.
         The combined frame_range covers both.
      3. After each absorption, re-run the same-state merge from Stage 2,
         because absorbing a segment may have brought two same-state blocks
         into contact.

    Why the neighbour's state wins (not the short segment's)?

    The short segment is the *noise*. If a 1-interval "moving" blip sits between
    two "stationary" blocks, it almost certainly came from a jittery detection,
    not real motion. Letting the longer, more evidence-backed neighbour state
    override it is the right prior. If the short segment were genuinely important
    (a real brief movement), it would accumulate enough intervals to survive the
    threshold — or the user should lower min_segment_frames.

    Why merge into the shorter neighbour?

    Merging into the shorter neighbour minimises information loss: we're
    overwriting a smaller region of the timeline rather than a larger one.
    For boundary segments (no choice of neighbour) we use the only available one.
    """
    # Work on a deep-enough copy so we don't mutate the caller's list.
    segs: List[_Segment] = [
        {"frame_range": list(s["frame_range"]), "state": s["state"], "_n": s["_n"]}
        for s in segments
    ]

    changed = True
    while changed:
        changed = False

        for i in range(len(segs)):
            if segs[i]["_n"] >= min_segment_frames:
                continue  # this segment is long enough, skip it
            if len(segs) == 1:
                break      # lone segment: nothing to merge into, stop

            # ── Choose neighbour index ─────────────────────────────────────
            if i == 0:
                ni = 1
            elif i == len(segs) - 1:
                ni = i - 1
            else:
                left_n = segs[i - 1]["_n"]
                right_n = segs[i + 1]["_n"]
                # Prefer the shorter neighbour; left wins ties.
                ni = (i - 1) if left_n <= right_n else (i + 1)

            short = segs[i]
            neighbour = segs[ni]

            # ── Absorb short segment into neighbour ─────────────────────────
            # The combined range spans both; the neighbour's state is kept.
            new_start = min(short["frame_range"][0], neighbour["frame_range"][0])
            new_end   = max(short["frame_range"][1], neighbour["frame_range"][1])
            segs[ni] = {
                "frame_range": [new_start, new_end],
                "state": neighbour["state"],
                "_n": neighbour["_n"] + short["_n"],
            }
            segs.pop(i)
            changed = True
            break  # restart scan — indices are invalidated after pop()

        if changed:
            # Re-merge same-state neighbours that the absorption may have created.
            # (E.g. absorbing the short segment between two "stationary" blocks
            # leaves them adjacent and they must be collapsed into one.)
            clean: List[_Segment] = []
            for seg in segs:
                if clean and clean[-1]["state"] == seg["state"]:
                    clean[-1]["frame_range"][1] = seg["frame_range"][1]
                    clean[-1]["_n"] += seg["_n"]
                else:
                    clean.append({
                        "frame_range": list(seg["frame_range"]),
                        "state": seg["state"],
                        "_n": seg["_n"],
                    })
            segs = clean

    return segs
