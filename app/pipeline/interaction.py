"""Detects person-object interactions by analysing spatial proximity and motion correlation."""

# ──────────────────────────────────────────────────────────────────────────────
# What counts as an "interaction" in this pipeline?
#
# True hand-level interaction detection would require a pose estimator (e.g.
# MediaPipe Hands or OpenPose) to locate wrist/finger keypoints and check
# whether they physically touch the object. That is the production upgrade path.
#
# Here we use a pragmatic proxy: a person and an object are "interacting" when
# their YOLO bounding boxes are in sustained spatial contact — either overlapping
# or within a configurable pixel gap.
#
# Why is bbox proximity a reasonable proxy?
#
# A person manipulating a lab instrument (plugging a cable, adjusting a knob)
# leans close enough that their body bounding box (torso + arms) either overlaps
# or nearly touches the object's bounding box. The key word is "sustained":
# a single proximate frame could be the person walking past; several consecutive
# proximate frames indicate they have stopped and engaged.
#
# Limitations:
#   - A person working at a bench out of camera frame (e.g. reaching across)
#     can be proximate in 2-D even when not touching in 3-D.
#   - Camera angle affects apparent proximity (top-down vs. side view).
#   - The person bbox covers the whole body, not just the hands — a person
#     standing near a machine with arms at their sides would still trigger proximity.
#
# These limitations are documented in README.md ("Design Decisions").
# ──────────────────────────────────────────────────────────────────────────────

from math import sqrt
from typing import Any, Dict, List, Optional, Tuple

# Type aliases matching the rest of the pipeline.
BBox = List[float]              # [x1, y1, x2, y2]
Observation = Tuple[int, BBox]  # (frame_index, bbox)
TrackRecord = Dict[str, Any]    # as produced by detection.py


# ──────────────────────────────────────────────────────────────────────────────
# Public function
# ──────────────────────────────────────────────────────────────────────────────

def detect_interactions(
    person_tracks: Dict[int, TrackRecord],
    object_tracks: Dict[int, TrackRecord],
    proximity_threshold: float = 30.0,
    min_interaction_frames: int = 5,
    gap_tolerance: int = 5,
) -> Dict[int, List[Dict[str, Any]]]:
    """Find every sustained person-object proximity episode in a video.

    The function iterates over all (person, object) pairs, evaluates spatial
    contact on every frame where both tracks have an observation, converts the
    resulting boolean signal into intervals, and applies post-processing filters
    to remove noise.

    Parameters
    ----------
    person_tracks:
        Dict keyed by track_id for tracks where is_person == True.
        Each value is a TrackRecord with an "observations" key:
            [(frame_index, [x1, y1, x2, y2]), ...]

    object_tracks:
        Same structure, but for non-person tracks.

    proximity_threshold:
        Maximum pixel gap between nearest bounding-box edges for the pair
        to be considered "in contact". Default 30 pixels.

        Why combine overlap OR edge-proximity?

        Camera angle determines whether a person's bbox overlaps the object's
        bbox or merely sits adjacent to it. Viewed from the side, a person
        plugging a cable will have their torso bbox just to the left of the
        device bbox — gap of maybe 10–40 px depending on resolution, but no
        overlap. Viewed from above, the same action might produce full overlap.
        Using an edge-distance test covers both geometries with one threshold.

        Why not use centre-to-centre distance instead?

        Centre distance ignores box size. A nearby large object (a bench) and a
        tiny distant object could have the same centre distance as a small nearby
        object (a pipette). Edge distance normalises for size: it measures the
        actual gap between surfaces, not the gap between centroids.

    min_interaction_frames:
        Minimum raw-frame span (frame_end − frame_start) an interaction must
        cover to be kept. Interactions shorter than this are treated as spurious
        proximity and discarded.

        Why is this filter needed?

        Short phantom interactions arise from two sources:
          1. Brief "walking past" events — the person passes near the object for
             a fraction of a second. A 0.5 s window at 30 fps = 15 frames; a
             threshold of 5 requires at least ~0.17 s, which filters most
             walkthroughs at normal stride.
          2. Ghost tracks from ByteTrack — very short-lived tracks (<10 frames)
             that were confirmed then lost. If a ghost track briefly coincides
             with an object, the resulting interaction is spurious.

        Units: raw video frames. Scale with fps and frame_stride as appropriate.
        Example: 30 fps, stride=1 → threshold=15 requires ≥0.5 s of contact.

    gap_tolerance:
        Maximum frame gap between two contact intervals that can be bridged into
        a single continuous interaction. Default 5 frames.

        Why is this needed?

        YOLO occasionally misses a detection for 1–3 frames even when the object
        is clearly visible — the confidence score dips below the threshold due to
        motion blur, partial occlusion, or inference noise. Without bridging, one
        real 10-second interaction where the detector dropped 2 frames in the
        middle would appear as two 5-second interactions in the output. A
        gap_tolerance of 5 frames (≈0.17 s at 30 fps) bridges this without
        accidentally merging two genuinely separate interaction episodes.

        Units: raw video frames. This is stride-dependent — with stride=5, a
        gap_tolerance=5 means only a 1-frame observed dropout would be bridged.
        Consider scaling: gap_tolerance_effective ≈ gap_tolerance * stride.

    Returns
    -------
    Dict mapping object_track_id to a list of interaction dicts:
        {"interacted_by_person": person_track_id, "frame_start": int, "frame_end": int}
    Objects with no qualifying interactions are absent from the result.
    The dicts are shaped to match the Interaction Pydantic model exactly.
    """
    result: Dict[int, List[Dict[str, Any]]] = {}

    for obj_id, obj_track in object_tracks.items():

        # Build a frame-indexed lookup so we can O(1) retrieve the bbox for any
        # given frame. Dict comprehension over observations is O(n) once per track.
        obj_obs: Dict[int, BBox] = {
            frame: bbox for frame, bbox in obj_track["observations"]
        }

        obj_interactions: List[Dict[str, Any]] = []

        for person_id, person_track in person_tracks.items():

            person_obs: Dict[int, BBox] = {
                frame: bbox for frame, bbox in person_track["observations"]
            }

            # ── Find frames where BOTH tracks have an observation ─────────────
            #
            # Set intersection gives us exactly the frames we can meaningfully
            # evaluate. Frames where only one track was observed tell us nothing
            # about the spatial relationship between the pair.
            shared_frames = sorted(set(obj_obs.keys()) & set(person_obs.keys()))

            if not shared_frames:
                # Tracks never co-existed in the same frame — no interaction possible.
                continue

            # ── Evaluate spatial contact at each shared frame ─────────────────
            #
            # This is the core test. We build a per-frame boolean signal rather
            # than a list of contact frames because we need the shape of the
            # signal (which frames were NOT in contact) to correctly apply the
            # gap_tolerance later.
            contact_signal: Dict[int, bool] = {
                f: _in_contact(person_obs[f], obj_obs[f], proximity_threshold)
                for f in shared_frames
            }

            # ── Convert the boolean signal to contact intervals ───────────────
            raw_intervals = _boolean_signal_to_intervals(shared_frames, contact_signal)

            if not raw_intervals:
                continue

            # ── Bridge short gaps caused by detection dropout ─────────────────
            bridged = _bridge_gaps(raw_intervals, gap_tolerance)

            # ── Discard spurious short interactions ───────────────────────────
            # frame_end − frame_start gives the frame span (exclusive of the
            # start frame itself). An interaction at a single frame has span 0.
            kept = [
                (start, end) for start, end in bridged
                if end - start >= min_interaction_frames
            ]

            for start, end in kept:
                obj_interactions.append({
                    "interacted_by_person": person_id,
                    "frame_start": start,
                    "frame_end": end,
                })

        if obj_interactions:
            result[obj_id] = obj_interactions

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _box_edge_distance(bbox_a: BBox, bbox_b: BBox) -> float:
    """Compute the minimum Euclidean distance between the edges of two xyxy boxes.

    Returns 0.0 if the boxes overlap (including touching edges).

    How is it computed?

    For two axis-aligned rectangles, the nearest-edge distance decomposes cleanly
    into two independent 1-D gap calculations:

      dx = max(0, max(ax1, bx1) − min(ax2, bx2))
      dy = max(0, max(ay1, by1) − min(ay2, by2))

    Intuition for dx:
      - max(ax1, bx1) is the leftmost right-edge of the two boxes' left sides.
      - min(ax2, bx2) is the rightmost left-edge of the two boxes' right sides.
      - If the boxes overlap horizontally, this difference is negative → clamp to 0.
      - If box B is entirely to the right of box A, dx = bx1 − ax2 (the horizontal gap).

    Combining dx and dy with Euclidean distance gives the nearest-corner distance
    when the boxes are in a diagonal configuration, and the nearest-edge gap when
    one axis gap is zero (boxes are directly above/below or beside each other).

    Examples:
      Side-by-side with 20 px gap:   dx=20, dy=0  → distance = 20.0
      Stacked with 15 px gap:        dx=0,  dy=15 → distance = 15.0
      Diagonal, 10 px each axis:     dx=10, dy=10 → distance ≈ 14.1
      Overlapping:                   dx=0,  dy=0  → distance = 0.0
    """
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    # Horizontal gap: positive only when the boxes do not overlap horizontally.
    dx = max(0.0, max(ax1, bx1) - min(ax2, bx2))

    # Vertical gap: positive only when the boxes do not overlap vertically.
    dy = max(0.0, max(ay1, by1) - min(ay2, by2))

    # When both gaps are 0 the boxes overlap on both axes, meaning they intersect.
    # sqrt(0² + 0²) = 0, which is the correct return value.
    return sqrt(dx * dx + dy * dy)


def _in_contact(bbox_a: BBox, bbox_b: BBox, proximity_threshold: float) -> bool:
    """Return True if two bounding boxes overlap or are within proximity_threshold pixels.

    This unifies the overlap and edge-proximity tests:
      - Overlapping boxes have edge distance 0, and 0 <= any positive threshold.
      - Non-overlapping boxes are in contact if their edge gap ≤ threshold.

    By expressing both as a single distance test we avoid a separate overlap
    check and keep the logic in one place. The threshold is the only tunable
    parameter, and its unit (pixels) is the same as the bbox coordinates.
    """
    return _box_edge_distance(bbox_a, bbox_b) <= proximity_threshold


def _boolean_signal_to_intervals(
    shared_frames: List[int],
    contact_signal: Dict[int, bool],
) -> List[Tuple[int, int]]:
    """Convert a per-frame boolean contact signal into a list of (start, end) intervals.

    Walks the sorted shared_frames sequence and groups consecutive True entries
    into runs. A run ends when a False entry is encountered, regardless of how
    many video frames separate the two observation frames (the frame gap is
    handled later by _bridge_gaps, not here).

    Returns a list of (frame_start, frame_end) tuples for runs of True.
    frame_start and frame_end are actual video frame indices, inclusive.
    """
    intervals: List[Tuple[int, int]] = []
    run_start: Optional[int] = None
    run_end: Optional[int] = None

    for f in shared_frames:
        if contact_signal[f]:
            # Extend or start a contact run.
            if run_start is None:
                run_start = f
            run_end = f
        else:
            # Non-contact frame: close any open run.
            if run_start is not None:
                intervals.append((run_start, run_end))
                run_start = None
                run_end = None

    # Close a run that reached the end of the shared frames list.
    if run_start is not None:
        intervals.append((run_start, run_end))

    return intervals


def _bridge_gaps(
    intervals: List[Tuple[int, int]],
    gap_tolerance: int,
) -> List[Tuple[int, int]]:
    """Merge intervals whose frame gap is within gap_tolerance.

    Why merge sorted left-to-right rather than finding the smallest gap first?

    Left-to-right is O(n) and guarantees that every adjacent pair is evaluated
    exactly once. Because intervals are sorted by start frame, a merged interval
    never needs to be reconsidered with earlier intervals. A priority-queue
    approach would be O(n log n) with no benefit here.

    The gap between interval i and interval i+1 is:
        gap = interval[i+1].start − interval[i].end

    If gap <= gap_tolerance, the two intervals are merged into one whose end is
    interval[i+1].end. The merged interval may then be compared with interval[i+2]
    in the same linear pass — the while loop handles cascading merges.
    """
    if not intervals:
        return []

    # Use a list of mutable lists so we can update the end of the last interval
    # in-place without rebuilding the whole list.
    merged: List[List[int]] = [list(intervals[0])]

    for start, end in intervals[1:]:
        gap = start - merged[-1][1]
        if gap <= gap_tolerance:
            # Bridge the gap: absorb this interval into the previous one.
            merged[-1][1] = end
        else:
            merged.append([start, end])

    return [(s, e) for s, e in merged]
