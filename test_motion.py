"""Smoke-test for motion classification — runs detection then prints each track's motion history.

Run from the project root (venv active):
    python test_motion.py                             # first video in data/uploads/
    python test_motion.py path/to/video.mp4           # explicit path
    python test_motion.py --stride 3                  # sample every 3rd frame
    python test_motion.py --threshold 3.5             # stricter motion threshold
    python test_motion.py --min-seg 5                 # require 5 intervals before keeping a state
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.pipeline.detection import DEFAULT_TRUSTED_CLASSES, Detector
from app.pipeline.motion import classify_motion


def find_first_video(upload_dir: Path) -> Optional[Path]:
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        matches = sorted(upload_dir.glob(ext))
        if matches:
            return matches[0]
    return None


def frames_to_seconds(frame: int, fps: float) -> float:
    return frame / fps if fps > 0 else 0.0


def state_bar(state: str, n_intervals: int) -> str:
    """Simple ASCII bar so states are scannable at a glance."""
    symbol = ">" if state == "moving" else "-"
    width = min(n_intervals * 3, 40)  # cap bar width at 40 chars
    return symbol * max(width, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test motion classification.")
    parser.add_argument("video", nargs="?",
                        help="Video path; defaults to first file in data/uploads/.")
    parser.add_argument("--stride", type=int, default=1,
                        help="frame_stride for detection (default: 1).")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="YOLO confidence threshold (default: 0.5).")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="motion_threshold in px/frame (default: 2.0).")
    parser.add_argument("--min-seg", type=int, default=3,
                        help="min_segment_frames for smoothing (default: 3).")
    parser.add_argument("--trusted", type=str, default=None,
                        help="Comma-separated trusted COCO classes (default: 'person').")
    args = parser.parse_args()

    # ── Resolve video ─────────────────────────────────────────────────────────
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"[ERROR] File not found: {video_path}")
            sys.exit(1)
    else:
        upload_dir = Path(__file__).parent / "data" / "uploads"
        video_path = find_first_video(upload_dir)
        if video_path is None:
            print(f"[ERROR] No video files found in {upload_dir}")
            print("  Upload a video via POST /tasks or pass a path directly.")
            sys.exit(1)

    trusted_classes = (
        {c.strip() for c in args.trusted.split(",")}
        if args.trusted else DEFAULT_TRUSTED_CLASSES
    )

    print(f"Video            : {video_path}")
    print(f"Stride           : every {args.stride} frame(s)")
    print(f"Conf threshold   : {args.conf}")
    print(f"Motion threshold : {args.threshold} px/frame")
    print(f"Min seg intervals: {args.min_seg}")
    print(f"Trusted classes  : {sorted(trusted_classes)}")
    print()

    # ── Detect + track ────────────────────────────────────────────────────────
    print("Running detection + tracking (this may take a moment)...")
    detector = Detector(
        confidence_threshold=args.conf,
        trusted_classes=trusted_classes,
    )
    metadata, tracks = detector.detect_and_track(str(video_path), frame_stride=args.stride)
    fps = metadata["fps"]

    # ── Print video metadata ──────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("VIDEO METADATA")
    print("=" * 65)
    print(f"  {metadata['resolution']}  |  {metadata['fps']:.1f} fps  |  "
          f"{metadata['duration']:.1f} s  |  {metadata['frame_count']} frames")

    if not tracks:
        print("\n  No tracks detected.")
        return

    # ── Classify motion for each track, then display ──────────────────────────
    print()
    print("=" * 65)
    print("MOTION HISTORY PER TRACK")
    print("=" * 65)
    print()
    print("  Legend:  >>>  moving     ---  stationary")
    print()

    for track_id, record in sorted(tracks.items()):
        obs = record["observations"]
        label = record["class_label"]
        raw = record["raw_coco_label"]
        coco_hint = f"  (COCO guess: {raw})" if raw != label else ""

        # ── Classify this track's motion ──────────────────────────────────
        history = classify_motion(
            obs,
            motion_threshold=args.threshold,
            min_segment_frames=args.min_seg,
        )

        # ── Track header ──────────────────────────────────────────────────
        first_frame = obs[0][0] if obs else 0
        last_frame  = obs[-1][0] if obs else 0
        duration_s  = frames_to_seconds(last_frame - first_frame, fps)
        person_tag  = "  [PERSON]" if record["is_person"] else ""

        print(f"  Track {track_id:>3d}  |  {label:15s}{coco_hint}")
        print(f"           |  {len(obs)} observations  |  "
              f"frames {first_frame}–{last_frame}  |  {duration_s:.1f} s{person_tag}")

        if not history:
            print("           |  (no motion history — no observations)")
            print()
            continue

        # ── Per-segment lines ─────────────────────────────────────────────
        for seg in history:
            start_f, end_f = seg["frame_range"]
            state = seg["state"]
            seg_dur_s = frames_to_seconds(end_f - start_f, fps)

            # How many observation-to-observation intervals fell in this segment?
            # Count from the observations that land inside [start_f, end_f].
            n_obs_in_seg = sum(
                1 for (f, _) in obs if start_f <= f <= end_f
            )
            n_intervals = max(n_obs_in_seg - 1, 0)

            bar = state_bar(state, n_intervals)
            label_col = f"  {state.upper():12s}" if state == "moving" else f"  {state:12s}"
            print(
                f"           |  {label_col}  "
                f"frames {start_f:5d}–{end_f:5d}  "
                f"({seg_dur_s:5.1f} s)  {bar}"
            )

        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  {'ID':>4}  {'label':15}  {'segments':>8}  {'moving%':>8}  {'note'}")
    print(f"  {'--':>4}  {'-----':15}  {'--------':>8}  {'-------':>8}")

    for track_id, record in sorted(tracks.items()):
        obs = record["observations"]
        history = classify_motion(obs, motion_threshold=args.threshold,
                                   min_segment_frames=args.min_seg)

        if not history:
            continue

        total_span = history[-1]["frame_range"][1] - history[0]["frame_range"][0]
        moving_span = sum(
            s["frame_range"][1] - s["frame_range"][0]
            for s in history if s["state"] == "moving"
        )
        pct = (moving_span / total_span * 100) if total_span > 0 else 0.0

        note = ""
        if record["is_person"]:
            note = "<-- person (expect some movement)"
        elif pct < 5:
            note = "(mostly stationary -- likely equipment)"
        elif pct > 80:
            note = "(mostly moving)"

        print(f"  {track_id:>4}  {record['class_label']:15}  "
              f"{len(history):>8}  {pct:>7.1f}%  {note}")

    print()


if __name__ == "__main__":
    main()
