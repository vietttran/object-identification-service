"""
pytest tests for app/pipeline/interaction.py — detect_interactions and helper functions.

Tests are pure-logic: they build synthetic track dicts (no video, no YOLO) and feed
them directly to the functions under test. This makes the suite fast, deterministic,
and runnable without any GPU or model weights.

Test coverage:
  • _box_edge_distance: geometry helper (overlapping, side-by-side, stacked, diagonal)
  • detect_interactions: happy path, negative (no contact), proximity, duration filter,
    gap bridging, and gap rejection.

Run:
    pytest tests/test_interaction.py -v
"""

import pytest
from math import sqrt

from app.pipeline.interaction import (
    _box_edge_distance,
    _boolean_signal_to_intervals,
    _bridge_gaps,
    _in_contact,
    detect_interactions,
)


# ---------------------------------------------------------------------------
# Helpers — build the TrackRecord dicts that detect_interactions expects
# ---------------------------------------------------------------------------

def _person(observations):
    """Build a person TrackRecord (is_person=True) from a list of (frame, bbox) tuples."""
    return {
        "class_label": "person",
        "raw_coco_label": "person",
        "is_person": True,
        "observations": observations,
    }


def _object(observations, label="object"):
    """Build a non-person TrackRecord from a list of (frame, bbox) tuples."""
    return {
        "class_label": label,
        "raw_coco_label": label,
        "is_person": False,
        "observations": observations,
    }


def _bbox(x1, y1, x2, y2):
    """Shorthand for an xyxy bounding box."""
    return [float(x1), float(y1), float(x2), float(y2)]


# ---------------------------------------------------------------------------
# _box_edge_distance — geometry of nearest-edge distance
# ---------------------------------------------------------------------------

def test_box_edge_distance_overlapping_boxes_returns_zero():
    """
    When two bounding boxes overlap (or share an edge), the minimum edge-to-edge
    distance is 0.0. This is the foundational check: if overlapping boxes don't
    return 0, the entire proximity logic is broken.

    Why does overlap → 0?
    The formula is max(0, max(ax1,bx1) − min(ax2,bx2)). When boxes overlap
    horizontally, max(ax1,bx1) < min(ax2,bx2) → negative clamped to 0. Same for y.
    sqrt(0² + 0²) = 0.0.
    """
    box_a = _bbox(0, 0, 100, 100)
    box_b = _bbox(50, 50, 150, 150)   # overlaps box_a in the bottom-right quadrant

    assert _box_edge_distance(box_a, box_b) == pytest.approx(0.0)


def test_box_edge_distance_touching_edges_returns_zero():
    """
    Two boxes that share exactly one edge (touching, not overlapping) also have
    edge distance 0. This matters because _in_contact uses ≤, not <, so a person
    standing right next to an object counts as contact at threshold=0.
    """
    box_a = _bbox(0, 0, 100, 100)
    box_b = _bbox(100, 0, 200, 100)   # shares the right edge of box_a

    assert _box_edge_distance(box_a, box_b) == pytest.approx(0.0)


def test_box_edge_distance_side_by_side_returns_horizontal_gap():
    """
    Two non-overlapping boxes separated only horizontally: the distance is the
    pure horizontal gap (dy = 0 because y-ranges overlap).

    Example used in the interaction.py docstring: 'Side-by-side with 20 px gap → 20.0'
    We use 30 px to test a different value.
    """
    box_a = _bbox(0,   0, 100, 100)
    box_b = _bbox(130, 0, 230, 100)   # 30 px horizontal gap (130 − 100 = 30)

    assert _box_edge_distance(box_a, box_b) == pytest.approx(30.0)


def test_box_edge_distance_stacked_returns_vertical_gap():
    """
    Two non-overlapping boxes separated only vertically: the distance equals
    the pure vertical gap (dx = 0 because x-ranges overlap).

    Models a lab bench (wide, low box) directly below a screen (wide, higher box)
    with a gap between them.
    """
    box_a = _bbox(0, 0,   100, 100)
    box_b = _bbox(0, 120, 100, 220)   # 20 px vertical gap (120 − 100 = 20)

    assert _box_edge_distance(box_a, box_b) == pytest.approx(20.0)


def test_box_edge_distance_diagonal_returns_euclidean_corner_distance():
    """
    Two boxes separated diagonally: the nearest-edge distance equals the Euclidean
    distance between the nearest two corners (one corner from each box).

    Example: both axes have a 30 px gap → sqrt(30² + 30²) ≈ 42.43.

    This is the geometry in the interaction.py docstring ('Diagonal, 10 px each axis
    → ≈14.1'). We use 30 px per axis for a cleaner mental-math check.
    """
    box_a = _bbox(0,   0,   100, 100)
    box_b = _bbox(130, 130, 230, 230)   # 30 px gap in both x and y

    expected = sqrt(30.0 ** 2 + 30.0 ** 2)
    assert _box_edge_distance(box_a, box_b) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# detect_interactions — main function
# ---------------------------------------------------------------------------

def test_sustained_overlap_produces_one_interaction():
    """
    A person and object whose boxes overlap for many consecutive frames should
    yield exactly one interaction entry with the correct frame span.

    This is the happy-path test: it verifies the full pipeline from shared-frame
    detection through to the final output dict structure.

    Setup: person and object share 20 frames (0–19) with fully overlapping boxes.
    proximity_threshold=0.0 (only count true overlaps), min_interaction_frames=5,
    gap_tolerance=0.
    """
    shared_bbox = _bbox(0, 0, 100, 100)
    frames = list(range(20))

    person_tracks = {1: _person([(f, shared_bbox) for f in frames])}
    object_tracks = {2: _object([(f, shared_bbox) for f in frames])}

    result = detect_interactions(
        person_tracks,
        object_tracks,
        proximity_threshold=0.0,
        min_interaction_frames=5,
        gap_tolerance=0,
    )

    assert 2 in result, "Object 2 should have at least one interaction"
    interactions = result[2]
    assert len(interactions) == 1

    iv = interactions[0]
    assert iv["interacted_by_person"] == 1
    assert iv["frame_start"] == 0
    assert iv["frame_end"] == 19


def test_far_apart_boxes_produce_no_interaction():
    """
    Person and object separated by more than proximity_threshold should yield
    no interaction entry in the result.

    Setup: person at [0,0,100,100], object at [500,0,600,100] — 400 px apart.
    proximity_threshold=30 → gap (400 px) >> threshold → no contact.

    This verifies that the proximity check actually rejects distant boxes, not
    that it always accepts everything.
    """
    person_tracks = {1: _person([(f, _bbox(0,   0, 100, 100)) for f in range(20)])}
    object_tracks = {2: _object([(f, _bbox(500, 0, 600, 100)) for f in range(20)])}

    result = detect_interactions(
        person_tracks,
        object_tracks,
        proximity_threshold=30.0,
        min_interaction_frames=5,
        gap_tolerance=0,
    )

    # Object 2 should be absent from the result (no qualifying interactions).
    assert 2 not in result


def test_boxes_within_proximity_threshold_but_not_overlapping_produce_interaction():
    """
    Person and object that do NOT overlap but are within proximity_threshold of each
    other's edges should still qualify as 'in contact'.

    This tests the edge-proximity part of _in_contact (as opposed to the overlap
    case tested above). It models a person standing right next to a lab instrument
    without their bbox actually covering it.

    Setup: object at [0,0,100,100]; person at [120,0,220,100] — 20 px horizontal gap.
    proximity_threshold=30 → gap (20 px) ≤ 30 → contact.
    """
    person_tracks = {
        1: _person([(f, _bbox(120, 0, 220, 100)) for f in range(15)])
    }
    object_tracks = {
        2: _object([(f, _bbox(0,   0, 100, 100)) for f in range(15)])
    }

    result = detect_interactions(
        person_tracks,
        object_tracks,
        proximity_threshold=30.0,   # 20 px gap < 30 threshold → in contact
        min_interaction_frames=5,
        gap_tolerance=0,
    )

    assert 2 in result
    assert result[2][0]["interacted_by_person"] == 1


def test_brief_contact_filtered_by_min_interaction_frames():
    """
    An interaction whose frame span is shorter than min_interaction_frames must
    be discarded from the output.

    Why does this filter exist?
    A person walking past an object generates a brief proximity blip (maybe 2–4
    frames) that is NOT a meaningful interaction. min_interaction_frames requires
    sustained contact so that walk-bys don't pollute the result.

    Setup: person and object overlap for only 3 frames (span = 3).
    min_interaction_frames=5 → 3 < 5 → interaction is rejected.
    """
    frames = list(range(4))   # frames 0,1,2,3 → span = 3 - 0 = 3

    person_tracks = {1: _person([(f, _bbox(0, 0, 100, 100)) for f in frames])}
    object_tracks = {2: _object([(f, _bbox(0, 0, 100, 100)) for f in frames])}

    result = detect_interactions(
        person_tracks,
        object_tracks,
        proximity_threshold=0.0,
        min_interaction_frames=5,   # requires span ≥ 5; we have 3 → filtered
        gap_tolerance=0,
    )

    assert 2 not in result, (
        "Interaction with span=3 should be discarded when min_interaction_frames=5"
    )


def test_gap_within_tolerance_bridged_into_one_interaction():
    """
    Two contact intervals separated by a gap ≤ gap_tolerance must be bridged
    into a single interaction.

    Why bridging? YOLO misses detections for 1–5 frames due to motion blur or
    confidence fluctuations. Without bridging, one 30-second real interaction
    with two dropped frames in the middle would appear as two 15-second events.
    gap_tolerance=5 bridges small detector dropouts.

    Setup:
      frames 0–10:  person and object overlap → contact interval [0, 10]
      frames 11–14: person moves far away → no contact (4-frame gap)
      frames 15–30: person and object overlap again → contact interval [15, 30]

    Gap = 15 − 10 = 5 ≤ gap_tolerance=5 → intervals are bridged → one interaction [0, 30].
    """
    near_bbox = _bbox(0,   0, 100, 100)
    far_bbox  = _bbox(500, 0, 600, 100)   # far enough to break proximity

    person_obs = (
        [(f, near_bbox) for f in range(11)] +     # frames 0–10: close
        [(f, far_bbox)  for f in range(11, 15)] + # frames 11–14: far (no contact)
        [(f, near_bbox) for f in range(15, 31)]   # frames 15–30: close again
    )
    object_obs = [(f, _bbox(0, 0, 100, 100)) for f in range(31)]

    person_tracks = {1: _person(person_obs)}
    object_tracks = {2: _object(object_obs)}

    result = detect_interactions(
        person_tracks,
        object_tracks,
        proximity_threshold=0.0,
        min_interaction_frames=5,
        gap_tolerance=5,   # gap of 5 frames is within tolerance → bridged
    )

    assert 2 in result
    interactions = result[2]
    assert len(interactions) == 1, "Two contact intervals with gap=5 should bridge into one"

    iv = interactions[0]
    assert iv["frame_start"] == 0
    assert iv["frame_end"] == 30


def test_gap_exceeds_tolerance_produces_two_separate_interactions():
    """
    Two contact intervals separated by more than gap_tolerance must NOT be
    bridged; they appear as two distinct interaction entries.

    This is the companion to the gap-bridging test above. If we increase the gap
    from 5 to 6 while keeping gap_tolerance=5, the bridge no longer applies and
    two separate interactions are returned.

    Both interactions must independently meet min_interaction_frames=5.

    Setup: same as above but gap is 6 frames (11–16), which exceeds gap_tolerance=5.
    """
    near_bbox = _bbox(0,   0, 100, 100)
    far_bbox  = _bbox(500, 0, 600, 100)

    person_obs = (
        [(f, near_bbox) for f in range(11)] +     # frames 0–10: contact interval [0,10]
        [(f, far_bbox)  for f in range(11, 17)] + # frames 11–16: gap of 6 frames
        [(f, near_bbox) for f in range(17, 31)]   # frames 17–30: contact interval [17,30]
    )
    object_obs = [(f, _bbox(0, 0, 100, 100)) for f in range(31)]

    person_tracks = {1: _person(person_obs)}
    object_tracks = {2: _object(object_obs)}

    result = detect_interactions(
        person_tracks,
        object_tracks,
        proximity_threshold=0.0,
        min_interaction_frames=5,
        gap_tolerance=5,   # gap of 6 > tolerance of 5 → NOT bridged
    )

    assert 2 in result
    interactions = result[2]
    assert len(interactions) == 2, (
        "Gap of 6 exceeds gap_tolerance=5; two separate interactions expected"
    )

    # Verify both interactions have the correct frame spans.
    starts = sorted(iv["frame_start"] for iv in interactions)
    ends   = sorted(iv["frame_end"]   for iv in interactions)
    assert starts == [0, 17]
    assert ends   == [10, 30]


def test_no_person_tracks_returns_empty_result():
    """
    detect_interactions with no person tracks should return {} immediately.

    Why test this? processor.py skips the detect_interactions call when there are
    no persons, but the function itself is also called with an empty dict from test
    scenarios. It must not crash on empty input.
    """
    object_tracks = {1: _object([(f, _bbox(0, 0, 100, 100)) for f in range(20)])}
    result = detect_interactions({}, object_tracks)
    assert result == {}


def test_non_overlapping_observation_windows_produce_no_interaction():
    """
    If person and object tracks have no shared frames (they never co-existed in
    the same frame), there can be no interaction.

    This hits the 'shared_frames is empty → continue' branch in detect_interactions.
    It models a scenario where the person left the scene before the object entered it.
    """
    # Person visible at frames 0–9; object visible at frames 20–29. No overlap.
    person_tracks = {1: _person([(f, _bbox(0, 0, 100, 100)) for f in range(10)])}
    object_tracks = {2: _object([(f, _bbox(0, 0, 100, 100)) for f in range(20, 30)])}

    result = detect_interactions(
        person_tracks,
        object_tracks,
        proximity_threshold=50.0,
    )

    assert 2 not in result
