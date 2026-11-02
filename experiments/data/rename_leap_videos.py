#!/usr/bin/env python3
"""
Rename leap-dataset videos from te00.mp4-style names to plain numeric names.

Input directory structure:
  <root>/<dataset>/{test,train}/te00.mp4

Output (in-place rename):
  <root>/<dataset>/{test,train}/0.mp4

Rules:
  - Strip all leading letters from the stem.
  - Strip leading zeros from the remaining numeric part (but keep at least "0").
  - Keep the original file extension.

Usage:
  python rename_leap_videos.py [--dry-run] [ROOT_DIR]

  ROOT_DIR defaults to /data/alenachao/data/leap-dataset/videos/
"""

import argparse
import re
import sys
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

STRIP_LETTERS_RE = re.compile(r"^[a-zA-Z]+")


def new_stem(stem: str) -> str:
    """Strip leading letters then leading zeros from the stem."""
    numeric = STRIP_LETTERS_RE.sub("", stem)
    if not numeric:
        raise ValueError(f"No numeric portion found in stem '{stem}'")
    # lstrip zeros but keep at least one digit
    return str(int(numeric))


def rename_videos(root: Path, dry_run: bool = False) -> None:
    renamed = 0
    skipped = 0
    errors = 0

    # root/<dataset>/{test,train}/*.mp4
    for video_path in sorted(root.glob("*/*/*.mp4")):
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        stem = video_path.stem
        try:
            target_stem = new_stem(stem)
        except ValueError as exc:
            print(f"  SKIP  {video_path.relative_to(root)}  ({exc})")
            skipped += 1
            continue

        target_path = video_path.with_name(target_stem + video_path.suffix)

        if target_path == video_path:
            print(f"  SAME  {video_path.relative_to(root)}")
            skipped += 1
            continue

        if target_path.exists():
            print(f"  CONFLICT  {video_path.relative_to(root)}  →  {target_path.name}  (target already exists, skipping)")
            errors += 1
            continue

        action = "DRY-RUN" if dry_run else "RENAME"
        print(f"  {action}  {video_path.relative_to(root)}  →  {target_path.name}")

        if not dry_run:
            video_path.rename(target_path)
            renamed += 1
        else:
            renamed += 1  # count as "would rename" in dry-run

    label = "Would rename" if dry_run else "Renamed"
    print(f"\n{label}: {renamed}  |  Skipped: {skipped}  |  Errors: {errors}")


def main() -> None:
    default_root = "/data/alenachao/data/leap-dataset/videos"

    parser = argparse.ArgumentParser(description="Rename leap-dataset videos to numeric filenames.")
    parser.add_argument("root", nargs="?", default=default_root, help=f"Root videos directory (default: {default_root})")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be renamed without actually renaming.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: '{root}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    print(f"{'DRY RUN — ' if args.dry_run else ''}Scanning {root}\n")
    rename_videos(root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
