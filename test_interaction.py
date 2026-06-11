"""Smoke-test for interaction detection — runs detection then prints every person-object episode.

Run from the project root (venv active):
    python test_interaction.py                            # first video in data/uploads/
    python test_interaction.py path/to/video.mp4          # explicit path
    python test_interaction.py --stride 3                 # sample every 3rd frame
    python test_interaction.py --proximity 50             # wider proximity window
    python test_interaction.py --min-frames 15            # require 15-frame minimum
    python test_interaction.py --gap 10                   # bridge gaps up to 10 frames
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.pipeline.detection import DEFAULT_TRUSTED_CLASSES, Detector
from app.pipeline.interaction import detect_interactions


def find_first_video(upload_dir: Path) -> Optional[Path]:
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        matches = sorted(upload_dir.glob(ext))
        if matches:
            return matches[0]
    return None


def frames_to_seconds(n_frames: int, fps: float) -> float:
    return n_frames / fps if fps > 0 else 0.0


def timeline_bar(frame_start: int, frame_end: int, video_frames: int, width: int = 40) -> str:
    """Render a one-line ASCII timeline bar showing where the interaction sits in the video.

    |----[=====]------------| means the interaction occupies the middle fifth
    of the video. Useful for a quick eyeball: does it land where the cable
    plugging visually happens in the footage?
    """
    if video_frames <= 0:
        return ""
    left  = int(frame_start / video_frames * width)
    right = int(frame_end   / video_frames * width)
    right = max(right, left + 1)  # at least one character wide

    bar = "-" * left + "=" * (right - left) + "-" * (width - right)
    return f"|{bar}|"


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test interaction detection.")
    parser.add_argument("video", nargs="?",
                        help="Video path; defaults to first file in data/uploads/.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--trusted", type=str, default=None,
                        help="Comma-separated trusted COCO classes (default: 'person').")
    parser.add_argument("--proximity", type=float, default=30.0,
                        help="Edge-distance threshold in pixels (default: 30).")
    parser.add_argument("--min-frames", type=int, default=5,
                        help="Minimum interaction frame span (default: 5).")
    parser.add_argument("--gap", type=int, default=5,
                        help="Gap-tolerance in frames for bridging dropouts (default: 5).")
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
    print(f"Proximity        : {args.proximity} px")
    print(f"Min interaction  : {args.min_frames} frames")
    print(f"Gap tolerance    : {args.gap} frames")
    print(f"Trusted classes  : {sorted(trusted_classes)}")
    print()

    # ── Detect + track ────────────────────────────────────────────────────────
    print("Running detection + tracking...")
    detector = Detector(confidence_threshold=args.conf, trusted_classes=trusted_classes)
    metadata, tracks = detector.detect_and_track(str(video_path), frame_stride=args.stride)
    fps = metadata["fps"]
    total_frames = metadata["frame_count"]

    print(f"  Found {len(tracks)} total tracks.")

    # ── Split tracks into people and objects ──────────────────────────────────
    person_tracks = {tid: r for tid, r in tracks.items() if r["is_person"]}
    object_tracks = {tid: r for tid, r in tracks.items() if not r["is_person"]}

    print(f"  {len(person_tracks)} person track(s), {len(object_tracks)} object track(s).")

    if not person_tracks:
        print("\n  No person tracks detected — cannot find interactions.")
        print("  Tips: lower --conf, or try a different video.")
        return

    if not object_tracks:
        print("\n  No object tracks detected.")
        return

    print()

    # ── Run interaction detection ─────────────────────────────────────────────
    interactions = detect_interactions(
        person_tracks=person_tracks,
        object_tracks=object_tracks,
        proximity_threshold=args.proximity,
        min_interaction_frames=args.min_frames,
        gap_tolerance=args.gap,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print("=" * 65)
    print("INTERACTION DETECTION RESULTS")
    print("=" * 65)
    print()

    total_interactions = 0

    for obj_id, obj_track in sorted(object_tracks.items()):
        label     = obj_track["class_label"]
        raw_label = obj_track["raw_coco_label"]
        coco_hint = f"  (COCO guess: {raw_label})" if raw_label != label else ""

        obj_obs   = obj_track["observations"]
        obj_first = obj_obs[0][0]  if obj_obs else 0
        obj_last  = obj_obs[-1][0] if obj_obs else 0

        print(f"  Object {obj_id:>3d}  |  {label}{coco_hint}")
        print(f"             |  frames {obj_first}–{obj_last}  "
              f"({frames_to_seconds(obj_last - obj_first, fps):.1f} s visible)")

        obj_results = interactions.get(obj_id, [])

        if not obj_results:
            print("             |  No interactions detected.")
            print()
            continue

        # Group by person so output is readable when multiple people appear.
        by_person: Dict[int, List[dict]] = {}
        for iv in obj_results:
            by_person.setdefault(iv["interacted_by_person"], []).append(iv)

        for person_id, person_ivs in sorted(by_person.items()):
            person_obs   = person_tracks[person_id]["observations"]
            person_first = person_obs[0][0]  if person_obs else 0
            person_last  = person_obs[-1][0] if person_obs else 0

            print(f"             |  Person {person_id} "
                  f"(visible frames {person_first}–{person_last}):")

            for n, iv in enumerate(sorted(person_ivs, key=lambda x: x["frame_start"]), 1):
                s, e = iv["frame_start"], iv["frame_end"]
                dur_s = frames_to_seconds(e - s, fps)
                bar   = timeline_bar(s, e, total_frames)

                # Qualitative hint to help eyeball plausibility.
                if dur_s >= 2.0:
                    hint = " <-- sustained (likely real interaction)"
                elif dur_s >= 0.5:
                    hint = " <-- brief"
                else:
                    hint = " <-- very brief"

                print(f"             |    [{n}] frames {s:5d}–{e:5d}  "
                      f"({dur_s:5.2f} s)  {bar}{hint}")

                total_interactions += 1

        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 65)
    n_active_objects = len(interactions)
    print(f"  {total_interactions} interaction(s) across {n_active_objects} object(s)")
    print(f"  {len(person_tracks)} person(s)  |  {len(object_tracks)} object(s)  |  "
          f"{metadata['duration']:.1f} s video  |  {fps:.0f} fps")

    if total_interactions == 0:
        print()
        print("  No interactions found. Try:")
        print(f"    --proximity {args.proximity * 2:.0f}   (double the proximity window)")
        print(f"    --min-frames {max(1, args.min_frames // 2)}  (halve the minimum duration)")
        print(f"    --gap {args.gap * 2}         (bridge longer detection dropouts)")

    print()


if __name__ == "__main__":
    main()
