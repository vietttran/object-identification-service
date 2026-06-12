"""
pytest tests for app/pipeline/motion.py — classify_motion function.

These tests use entirely synthetic observations (no video files, no YOLO) so they
run instantly and deterministically in any environment, including CI. Each test
constructs hand-crafted bounding-box sequences designed to hit a specific code path
in classify_motion and its three sub-stages: _measure_intervals, _merge_adjacent_states,
and _smooth_short_segments.

Run:
    pytest tests/test_motion.py -v
"""

import pytest
from app.pipeline.motion import classify_motion


# ---------------------------------------------------------------------------
# Helpers — shared across all test cases
# ---------------------------------------------------------------------------

def _bbox(x: float, y: float, w: float = 100.0, h: float = 100.0):
    """Build an xyxy bounding box [x1, y1, x2, y2] anchored at (x, y).
    Centre = (x + w/2, y + h/2). All motion tests use the default 100×100 size
    so that box-size changes don't affect the centre calculation.
    """
    return [x, y, x + w, y + h]


def _obs(frame: int, x: float, y: float):
    """Shorthand for (frame_index, bbox) — the format detect_and_track produces."""
    return (frame, _bbox(x, y))


# ---------------------------------------------------------------------------
# Edge cases — inputs that bypass the main algorithm
# ---------------------------------------------------------------------------

def test_empty_observations_returns_empty_list():
    """
    classify_motion([]) must return [] without crashing.

    Why does this matter?
    processor.py calls classify_motion for every surviving track. The tracker
    can produce a track ID that was confirmed but then immediately lost — the
    resulting observations list is empty. An empty return is the correct signal
    to processor.py that there is nothing to include in the motion_history field.
    """
    assert classify_motion([]) == []


def test_single_observation_returns_one_stationary_segment():
    """
    With exactly one observation there is no consecutive pair to measure
    displacement on. The function returns a single 'stationary' segment
    covering the one observed frame.

    Why 'stationary' and not 'unknown'?
    Zero measured displacement is indistinguishable from a perfectly still object.
    Defaulting to 'stationary' is the conservative choice: calling something
    stationary that was actually moving once is safer (for the interviewer) than
    generating phantom motion events.

    Why must frame_range be [7, 7] (both endpoints equal)?
    The frame_range represents the time window. A single observation is a
    zero-duration window — the object was seen at frame 7, period. [7, 7] is
    the correct degenerate interval for that.
    """
    result = classify_motion([_obs(7, 50.0, 50.0)])

    assert len(result) == 1
    assert result[0]["state"] == "stationary"
    assert result[0]["frame_range"] == [7, 7]


# ---------------------------------------------------------------------------
# Pure-state sequences — only one state across all frames
# ---------------------------------------------------------------------------

def test_stationary_object_produces_single_stationary_segment():
    """
    An object whose bbox centre moves less than motion_threshold px/frame
    across all consecutive observation pairs should produce exactly one
    'stationary' segment spanning all observed frames.

    Setup: centre drifts 0.5 px horizontally per frame (well below threshold=2.0).
    This models real YOLO bounding-box jitter on a perfectly still lab instrument.
    """
    # 10 frames; bbox x shifts +0.5 per frame → centre moves 0.5 px/frame < 2.0.
    observations = [_obs(i, 100.0 + i * 0.5, 100.0) for i in range(10)]

    result = classify_motion(observations, motion_threshold=2.0, min_segment_frames=1)

    assert len(result) == 1
    assert result[0]["state"] == "stationary"
    # First and last frame of the segment must match the first and last observations.
    assert result[0]["frame_range"][0] == 0
    assert result[0]["frame_range"][1] == 9


def test_moving_object_produces_single_moving_segment():
    """
    An object whose centre consistently moves well above motion_threshold px/frame
    should produce exactly one 'moving' segment across all observations.

    Setup: centre moves 20 px/frame horizontally — 10× above the default threshold.
    This models a person walking across the lab.
    """
    # 8 frames; bbox x shifts +20 per frame → centre moves 20 px/frame >> 2.0.
    observations = [_obs(i, float(i * 20), 0.0) for i in range(8)]

    result = classify_motion(observations, motion_threshold=2.0, min_segment_frames=1)

    assert len(result) == 1
    assert result[0]["state"] == "moving"
    assert result[0]["frame_range"] == [0, 7]


# ---------------------------------------------------------------------------
# State transition — moving → stationary
# ---------------------------------------------------------------------------

def test_moving_then_stop_produces_two_segments_in_correct_order():
    """
    An object that moves for several frames then stops should yield exactly
    two segments: 'moving' first, then 'stationary'.

    Setup:
      frames 0–4: bbox shifts 20 px/frame → centre moves 20 px/frame → moving.
                  At frame 4 the object reaches x=80 and stops.
      frames 5–8: bbox stays at x=80 → centre is stationary (0 px/frame).

    Intervals breakdown:
      [0,1],[1,2],[2,3],[3,4]: 20 px/frame → moving  (4 intervals, _n=4)
      [4,5],[5,6],[6,7],[7,8]:  0 px/frame → stationary (4 intervals, _n=4)

    Both _n=4 ≥ min_segment_frames=3, so smoothing leaves them untouched.
    frame 4 appears as the shared boundary (end of moving, start of stationary).
    """
    moving_obs     = [_obs(i, float(i * 20), 0.0) for i in range(5)]   # frames 0–4
    stationary_obs = [_obs(i, 80.0, 0.0) for i in range(5, 9)]          # frames 5–8
    observations   = moving_obs + stationary_obs

    result = classify_motion(observations, motion_threshold=2.0, min_segment_frames=3)

    assert len(result) == 2
    # First segment covers the moving phase.
    assert result[0]["state"] == "moving"
    # Second segment covers the stationary phase.
    assert result[1]["state"] == "stationary"
    # The frame ranges must not overlap beyond the shared boundary frame.
    assert result[0]["frame_range"] == [0, 4]
    assert result[1]["frame_range"] == [4, 8]
    # Coverage: first observation → last observation, no gaps.
    assert result[0]["frame_range"][0] == 0   # starts at first observed frame
    assert result[1]["frame_range"][1] == 8   # ends at last observed frame


# ---------------------------------------------------------------------------
# Per-frame displacement normalisation — stride independence
# ---------------------------------------------------------------------------

def test_per_frame_normalisation_stride1_and_stride5_classify_identically():
    """
    The same physical motion speed must produce the same classification regardless
    of the sampling stride used during detection.

    Why does this matter?
    processor.py passes frame_stride as a parameter. A user might run with stride=5
    to speed up processing. classify_motion normalises displacement by the frame gap
    (per_frame_disp = disp / (frame_j − frame_i)), so a motion_threshold of 2.0 means
    '2 pixels per video frame' in both cases.

    Setup:
      stride-1: 5 obs at consecutive frames, x shifts 10 px each frame.
                per-frame displacement = 10 px → should be 'moving'.
      stride-5: 5 obs every 5th frame, x shifts 50 px (= 5 × 10) each observation.
                per-frame displacement = 50 / 5 = 10 px → should also be 'moving'.

    Without normalisation: stride-5 would score 50 px (25× above threshold) while
    stride-1 scores 10 px — same motion, wildly different thresholding behaviour.
    Normalisation fixes this by dividing by the actual frame gap.
    """
    # Stride-1: observations at frames 0,1,2,3,4 with 10 px/frame shift.
    obs_stride1 = [_obs(i, float(i * 10), 0.0) for i in range(5)]

    # Stride-5: observations at frames 0,5,10,15,20 with 50 px per 5-frame gap.
    obs_stride5 = [_obs(i * 5, float(i * 50), 0.0) for i in range(5)]

    result1 = classify_motion(obs_stride1, motion_threshold=2.0, min_segment_frames=1)
    result5 = classify_motion(obs_stride5, motion_threshold=2.0, min_segment_frames=1)

    # Both represent 10 px/frame > 2.0 threshold → both are single 'moving' segments.
    assert len(result1) == 1 and result1[0]["state"] == "moving"
    assert len(result5) == 1 and result5[0]["state"] == "moving"

    # The state label must match: same motion, same classification.
    assert result1[0]["state"] == result5[0]["state"]


# ---------------------------------------------------------------------------
# Short-segment smoothing
# ---------------------------------------------------------------------------

def test_smoothing_absorbs_brief_moving_blip_between_stationary_periods():
    """
    A 2-interval 'moving' blip sandwiched between two longer 'stationary' periods
    must be absorbed into the surrounding stationary state when min_segment_frames=3.

    This models real lab footage where YOLO jitters for 2 frames on a still object,
    or a person makes a small accidental hand twitch that barely crosses the threshold.
    Without smoothing, the output contains a spurious 'moving' entry that would
    generate a false interaction event downstream.

    Setup (all observations 5 frames apart):
      frames 0,5,10,15: at STILL position → 3 stationary intervals   (_n=3)
      frame 20:         at SPIKE position → intervals [15,20] and [20,25] are
                                            'moving' because displacement > threshold
      frames 25,30,35,40,45: back at STILL → 4 more stationary intervals (_n=4)

    Raw segments before smoothing:
      [0 ,15] stationary  (_n=3)
      [15,25] moving      (_n=2)  ← below threshold of 3 → gets absorbed
      [25,45] stationary  (_n=4)

    After absorption:
      [15,25] merges into its shorter neighbour [0,15] (left wins because 3 ≤ 4).
      The two resulting stationary blocks [0,25] and [25,45] merge into [0,45].
    """
    STILL = [100.0, 100.0, 200.0, 200.0]  # centre (150, 150)
    SPIKE = [150.0, 100.0, 250.0, 200.0]  # centre (200, 150) — 50 px away

    observations = (
        [(f, STILL) for f in [0, 5, 10, 15]] +
        [(20, SPIKE)] +
        [(f, STILL) for f in [25, 30, 35, 40, 45]]
    )

    result = classify_motion(observations, motion_threshold=2.0, min_segment_frames=3)

    assert len(result) == 1, (
        "The 2-interval moving blip should be absorbed; output should be one 'stationary' segment"
    )
    assert result[0]["state"] == "stationary"
    assert result[0]["frame_range"] == [0, 45]


def test_without_smoothing_brief_blip_is_visible():
    """
    With min_segment_frames=1 (no smoothing), the same brief blip that would be
    absorbed at min_segment_frames=3 must appear as a distinct 'moving' segment.

    This test acts as a contrast to test_smoothing_absorbs_brief_moving_blip:
    it confirms that the blip really exists in the raw signal and is only suppressed
    when smoothing is active. Both tests must pass for the smoothing logic to be
    correct — if only the smoothed case passes, we might have accidentally removed
    the blip at the measurement stage instead.
    """
    STILL = [100.0, 100.0, 200.0, 200.0]
    SPIKE = [150.0, 100.0, 250.0, 200.0]

    observations = (
        [(f, STILL) for f in [0, 5, 10, 15]] +
        [(20, SPIKE)] +
        [(f, STILL) for f in [25, 30, 35, 40, 45]]
    )

    result = classify_motion(observations, motion_threshold=2.0, min_segment_frames=1)

    # Should have 3 segments: stationary, then the blip, then stationary.
    assert len(result) == 3
    assert result[0]["state"] == "stationary"
    assert result[1]["state"] == "moving"   # the blip, frames 15–25
    assert result[2]["state"] == "stationary"


# ---------------------------------------------------------------------------
# Structural invariant
# ---------------------------------------------------------------------------

# Hand-crafted multi-phase sequences for the parametrised invariant test.
# Each produces multiple segments so there are adjacent pairs to check.
_STATIONARY_THEN_MOVING_THEN_STATIONARY = (
    # Phase 1: frames 0–4, perfectly still at x=0.
    [_obs(i, 0.0, 0.0) for i in range(5)]
    # Phase 2: frames 5–9, moving right 30 px/frame (large enough to be clearly 'moving').
    + [_obs(5 + j, 100.0 + j * 30, 0.0) for j in range(5)]
    # Phase 3: frames 10–14, still at the position the moving phase ended at (x=220).
    # Interval [9,10] is stationary (centre doesn't move between last moving and first still obs).
    + [_obs(10 + j, 220.0, 0.0) for j in range(5)]
)

_MOVING_THEN_STATIONARY = (
    [_obs(i, float(i * 20), 0.0) for i in range(5)]   # frames 0–4 moving
    + [_obs(i, 80.0, 0.0) for i in range(5, 10)]       # frames 5–9 stationary
)


@pytest.mark.parametrize("observations, label", [
    (
        [_obs(i, 0.0, 0.0) for i in range(5)],
        "all_stationary",
    ),
    (
        [_obs(i, float(i * 20), 0.0) for i in range(5)],
        "all_moving",
    ),
    (
        _MOVING_THEN_STATIONARY,
        "moving_then_stationary",
    ),
    (
        _STATIONARY_THEN_MOVING_THEN_STATIONARY,
        "stationary_moving_stationary",
    ),
])
def test_no_adjacent_segments_share_same_state(observations, label):
    """
    Invariant: in the output of classify_motion, no two adjacent segments ever
    have the same 'state'. This is guaranteed by _merge_adjacent_states (which
    run-length encodes same-state intervals) and is preserved by _smooth_short_segments
    (which re-merges adjacent same-state blocks after every absorption step).

    Why is this invariant important?
    If two adjacent segments both say 'stationary', the downstream JSON is
    redundant and confusing. The interval smoothing algorithm could also loop
    forever if it never collapsed adjacent same-state blocks after absorption.
    Verifying this invariant ensures the algorithm is self-consistent.

    We use min_segment_frames=1 to disable smoothing so that the invariant is
    tested at the merge-only level. The smoothed tests above verify it indirectly.
    """
    result = classify_motion(observations, motion_threshold=2.0, min_segment_frames=1)

    for i in range(len(result) - 1):
        assert result[i]["state"] != result[i + 1]["state"], (
            f"[{label}] Segments {i} and {i+1} both have state='{result[i]['state']}' — "
            "adjacent same-state segments should have been merged."
        )
