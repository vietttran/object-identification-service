"""Standalone smoke-test for the Detector class.

Run from the project root (with the venv active):
    python test_detector.py                          # first video in data/uploads/
    python test_detector.py path/to/video.mp4        # explicit path
    python test_detector.py --stride 3               # sample every 3rd frame
    python test_detector.py --conf 0.4               # lower confidence threshold
    python test_detector.py --trusted person,bottle  # trust extra COCO classes
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.pipeline.detection import DEFAULT_TRUSTED_CLASSES, Detector


def find_first_video(upload_dir: Path) -> Optional[Path]:
    """Return the first video file found in upload_dir, or None."""
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        matches = sorted(upload_dir.glob(ext))
        if matches:
            return matches[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the Detector.")
    parser.add_argument(
        "video",
        nargs="?",
        help="Path to a video file. Defaults to the first file in data/uploads/.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="frame_stride passed to detect_and_track (default: 1 = every frame).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold — detections below this score are dropped (default: 0.5).",
    )
    parser.add_argument(
        "--trusted",
        type=str,
        default=None,
        help=(
            "Comma-separated COCO class names to trust (default: 'person'). "
            "Example: --trusted person,bottle"
        ),
    )
    args = parser.parse_args()

    # ── Resolve video path ────────────────────────────────────────────────────
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
            print("  Upload a video via POST /tasks first, or pass a path directly.")
            sys.exit(1)

    # ── Parse trusted_classes ─────────────────────────────────────────────────
    trusted_classes = (
        {c.strip() for c in args.trusted.split(",")}
        if args.trusted
        else DEFAULT_TRUSTED_CLASSES
    )

    print(f"Video            : {video_path}")
    print(f"Stride           : every {args.stride} frame(s)")
    print(f"Conf threshold   : {args.conf}")
    print(f"Trusted classes  : {sorted(trusted_classes)}\n")

    # ── Run detection + tracking ──────────────────────────────────────────────
    detector = Detector(
        confidence_threshold=args.conf,
        trusted_classes=trusted_classes,
    )
    metadata, tracks = detector.detect_and_track(str(video_path), frame_stride=args.stride)

    # ── Video metadata ────────────────────────────────────────────────────────
    print("=" * 60)
    print("VIDEO METADATA")
    print("=" * 60)
    print(f"  Duration   : {metadata['duration']:.2f} s")
    print(f"  Frames     : {metadata['frame_count']}")
    print(f"  Resolution : {metadata['resolution']}")
    print(f"  FPS        : {metadata['fps']:.2f}")

    # ── Class breakdown ───────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"TRACKED OBJECTS  ({len(tracks)} unique confirmed tracks)")
    print("=" * 60)

    class_counts: dict = {}
    for record in tracks.values():
        label = record["class_label"]
        class_counts[label] = class_counts.get(label, 0) + 1

    print("  Authoritative label breakdown (what the pipeline uses):")
    for label, count in sorted(class_counts.items()):
        tag = " [people]" if label == "person" else ""
        print(f"    {label:25s}  {count:3d} track(s){tag}")

    # ── Relabeling summary ────────────────────────────────────────────────────
    # Show which COCO guesses were replaced with the generic "object" label.
    # This makes the OOD handling visible and verifiable without reading code.
    relabeled = [
        (tid, r["raw_coco_label"])
        for tid, r in tracks.items()
        if r["class_label"] == "object"
    ]
    if relabeled:
        print()
        print("  OOD relabeling — COCO guess -> 'object' (unreliable for this domain):")
        coco_guess_counts: dict = {}
        for _, raw in relabeled:
            coco_guess_counts[raw] = coco_guess_counts.get(raw, 0) + 1
        for raw_label, count in sorted(coco_guess_counts.items(), key=lambda x: -x[1]):
            print(f"    '{raw_label}' suppressed for {count} track(s)")
    else:
        print()
        print("  No OOD relabeling occurred (all detections were trusted classes).")

    # ── Per-track detail table ────────────────────────────────────────────────
    # The "label" column shows the authoritative class_label used downstream.
    # The "COCO guess" column shows raw_coco_label for debugging.
    # When they match, the COCO prediction was trusted. When they differ,
    # the COCO prediction was an OOD hallucination that was relabeled.
    print()
    hdr = f"  {'ID':>4}  {'label':15}  {'COCO guess':20}  {'obs':>5}  {'first':>7}  {'last':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for track_id, record in sorted(tracks.items()):
        obs = record["observations"]
        first_frame = obs[0][0] if obs else "-"
        last_frame  = obs[-1][0] if obs else "-"
        label       = record["class_label"]
        raw         = record["raw_coco_label"]
        # Flag rows where the label was changed so they're easy to spot.
        flag = " *" if label != raw else "  "
        print(
            f"  {track_id:>4}  {label:15}  {raw:20}  {len(obs):>5}"
            f"  {str(first_frame):>7}  {str(last_frame):>7}{flag}"
        )

    if relabeled:
        print("  (* = OOD relabeled: COCO guess was not in trusted_classes)")

    # ── People vs objects summary ─────────────────────────────────────────────
    print()
    people  = [tid for tid, r in tracks.items() if r["is_person"]]
    objects = [tid for tid, r in tracks.items() if not r["is_person"]]
    print(f"  Person track IDs : {people if people else '(none)'}")
    suffix = "..." if len(objects) > 10 else ""
    print(f"  Object track IDs : {objects[:10]}{suffix}")


if __name__ == "__main__":
    main()
