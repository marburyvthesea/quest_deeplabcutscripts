#!/usr/bin/env python3
"""Build a DLC training manifest from ezTrack output files.

The output manifest can be passed directly to
`dlc_build_training_project_from_tracking.py`.

Typical use on Quest:

    python dlc_build_manifest_from_eztrack.py \
      --find-under /scratch/jma819/behavCamData/YZ_linearTrackExperiments \
      --output lineartrack_manifest.csv

Or, for a curated subset of mice/sessions:

    python dlc_build_manifest_from_eztrack.py \
      --input \
      /scratch/jma819/behavCamData/YZ_linearTrackExperiments/BehavCamConcactenated_311/rotated_and_cropped_avi \
      /scratch/jma819/behavCamData/YZ_linearTrackExperiments/BehavCamConcactenated_326/rotated_and_cropped_avi \
      --output selected_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PairRecord:
    video_path: Path
    csv_path: Path
    source_dir: Path
    mouse_id: str
    clip_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "--paths",
        dest="paths",
        nargs="*",
        default=[],
        help=(
            "Directories or individual files to inspect. Files may be either "
            "the .avi or the matching _LocationOutput.csv. "
            "`--input` and `--paths` are interchangeable."
        ),
    )
    parser.add_argument(
        "--paths-file",
        default=None,
        help="Text file containing one directory or file path per line.",
    )
    parser.add_argument(
        "--find-under",
        nargs="*",
        default=[],
        help=(
            "Root directories to search recursively for folders named "
            "'rotated_and_cropped_avi'."
        ),
    )
    parser.add_argument(
        "--dir-name",
        default="rotated_and_cropped_avi",
        help="Folder name to search for under each --find-under root.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the output manifest CSV.",
    )
    parser.add_argument(
        "--video-extension",
        default=".avi",
        help="Video file extension to pair with each tracking CSV.",
    )
    parser.add_argument(
        "--csv-suffix",
        default="_LocationOutput.csv",
        help="Suffix used by ezTrack label CSV files.",
    )
    parser.add_argument(
        "--ignore-substring",
        action="append",
        default=["video_output"],
        help="Skip files whose names contain this substring. Repeat as needed.",
    )
    parser.add_argument(
        "--ignore-dir-name",
        action="append",
        default=["validation_videos"],
        help="Skip directories with this exact name. Repeat as needed.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error if any requested path is missing or unmatched.",
    )
    return parser.parse_args()


def load_paths_file(path: str | None) -> list[Path]:
    if not path:
        return []
    source = Path(path).expanduser().resolve()
    discovered: list[Path] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            discovered.append(Path(cleaned).expanduser().resolve())
    return discovered


def should_ignore_path(path: Path, args: argparse.Namespace) -> bool:
    if any(part in set(args.ignore_dir_name) for part in path.parts):
        return True
    return any(token in path.name for token in args.ignore_substring)


def discover_scan_targets(args: argparse.Namespace) -> list[Path]:
    targets = [Path(item).expanduser().resolve() for item in args.paths]
    targets.extend(load_paths_file(args.paths_file))

    for root in args.find_under:
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists():
            if args.strict:
                raise SystemExit(f"Search root does not exist: {root_path}")
            print(f"Warning: search root does not exist: {root_path}", file=sys.stderr)
            continue
        targets.extend(
            path.resolve()
            for path in root_path.rglob(args.dir_name)
            if path.is_dir() and not should_ignore_path(path, args)
        )

    unique_targets: list[Path] = []
    seen: set[Path] = set()
    for target in targets:
        if target not in seen:
            seen.add(target)
            unique_targets.append(target)
    return unique_targets


def infer_mouse_id(path: Path) -> str:
    for part in reversed(path.parts):
        match = re.search(r"BehavCamConcactenated_(\d+)", part)
        if match:
            return match.group(1)
    match = re.search(r"m(\d+)_", path.name)
    return match.group(1) if match else ""


def build_record_from_csv(csv_path: Path, args: argparse.Namespace) -> PairRecord | None:
    if should_ignore_path(csv_path, args):
        return None
    if not csv_path.name.endswith(args.csv_suffix):
        return None

    video_name = csv_path.name[: -len(args.csv_suffix)] + args.video_extension
    video_path = csv_path.with_name(video_name)
    if should_ignore_path(video_path, args):
        return None
    if not video_path.exists():
        return None

    return PairRecord(
        video_path=video_path.resolve(),
        csv_path=csv_path.resolve(),
        source_dir=csv_path.parent.resolve(),
        mouse_id=infer_mouse_id(csv_path),
        clip_name=video_path.stem,
    )


def build_record_from_video(video_path: Path, args: argparse.Namespace) -> PairRecord | None:
    if should_ignore_path(video_path, args):
        return None
    if video_path.suffix != args.video_extension:
        return None

    csv_name = video_path.stem + args.csv_suffix
    csv_path = video_path.with_name(csv_name)
    if should_ignore_path(csv_path, args):
        return None
    if not csv_path.exists():
        return None

    return PairRecord(
        video_path=video_path.resolve(),
        csv_path=csv_path.resolve(),
        source_dir=video_path.parent.resolve(),
        mouse_id=infer_mouse_id(video_path),
        clip_name=video_path.stem,
    )


def collect_pairs_from_directory(directory: Path, args: argparse.Namespace) -> tuple[list[PairRecord], list[Path]]:
    pairs: list[PairRecord] = []
    unmatched: list[Path] = []

    for csv_path in sorted(directory.glob(f"*{args.csv_suffix}")):
        if should_ignore_path(csv_path, args):
            continue
        record = build_record_from_csv(csv_path, args)
        if record is None:
            unmatched.append(csv_path)
            continue
        pairs.append(record)

    return pairs, unmatched


def collect_pairs(args: argparse.Namespace) -> tuple[list[PairRecord], list[Path], list[Path]]:
    targets = discover_scan_targets(args)
    pairs: list[PairRecord] = []
    unmatched: list[Path] = []
    missing_targets: list[Path] = []

    for target in targets:
        if not target.exists():
            missing_targets.append(target)
            continue

        if target.is_dir():
            found_pairs, unmatched_csvs = collect_pairs_from_directory(target, args)
            pairs.extend(found_pairs)
            unmatched.extend(unmatched_csvs)
            continue

        record = None
        if target.name.endswith(args.csv_suffix):
            record = build_record_from_csv(target, args)
        elif target.suffix == args.video_extension:
            record = build_record_from_video(target, args)

        if record is None:
            unmatched.append(target)
            continue
        pairs.append(record)

    deduped_pairs: list[PairRecord] = []
    seen_keys: set[tuple[Path, Path]] = set()
    for pair in sorted(pairs, key=lambda item: (str(item.video_path), str(item.csv_path))):
        key = (pair.video_path, pair.csv_path)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped_pairs.append(pair)

    return deduped_pairs, unmatched, missing_targets


def write_manifest(records: Iterable[PairRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video_path", "csv_path", "mouse_id", "clip_name", "source_dir"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "video_path": str(record.video_path),
                    "csv_path": str(record.csv_path),
                    "mouse_id": record.mouse_id,
                    "clip_name": record.clip_name,
                    "source_dir": str(record.source_dir),
                }
            )


def main() -> None:
    args = parse_args()
    pairs, unmatched, missing_targets = collect_pairs(args)

    if not pairs:
        raise SystemExit("No matched video/CSV pairs were found.")

    output_path = Path(args.output).expanduser().resolve()
    write_manifest(pairs, output_path)

    print(f"Wrote {len(pairs)} matched pairs to {output_path}")
    if missing_targets:
        print(f"Missing paths: {len(missing_targets)}", file=sys.stderr)
        for path in missing_targets[:10]:
            print(f"  {path}", file=sys.stderr)
    if unmatched:
        print(f"Unmatched paths: {len(unmatched)}", file=sys.stderr)
        for path in unmatched[:10]:
            print(f"  {path}", file=sys.stderr)

    if args.strict and (missing_targets or unmatched):
        raise SystemExit("Strict mode enabled and some paths were missing or unmatched.")


if __name__ == "__main__":
    main()
