#!/usr/bin/env python3
"""Build a DeepLabCut project from tracked X/Y CSV files and matching videos.

This script is meant for cases where an external tracker has already produced
frame-by-frame X/Y coordinates for a single object or body part. Given a CSV
manifest of video/annotation pairs, it will:

1. Create a new DeepLabCut project.
2. Update the project config with the requested single body part.
3. Sample frames from each video.
4. Extract those frames into the project's labeled-data folders.
5. Write DLC-style CollectedData CSV files from the tracked X/Y coordinates.
6. Convert those CSV files to H5 via DeepLabCut.

By default the script samples a manageable subset of frames from each video.
Use ``--all-frames`` if you truly want every labeled frame exported.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

os.environ.setdefault("DLClight", "True")


@dataclass(frozen=True)
class PairRecord:
    video_path: Path
    csv_path: Path

    @property
    def clip_name(self) -> str:
        return self.video_path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="CSV manifest with at least video_path and csv_path columns.",
    )
    parser.add_argument(
        "--project-name",
        required=True,
        help="DeepLabCut project name to create.",
    )
    parser.add_argument(
        "--experimenter",
        required=True,
        help="Experimenter/scorer name for the DLC project.",
    )
    parser.add_argument(
        "--working-directory",
        default=".",
        help="Directory where the DeepLabCut project should be created.",
    )
    parser.add_argument(
        "--bodypart",
        default="mouse_center",
        help="Single body part name to use for the tracked X/Y point.",
    )
    parser.add_argument(
        "--manifest-video-column",
        default="video_path",
        help="Manifest column containing video paths.",
    )
    parser.add_argument(
        "--manifest-csv-column",
        default="csv_path",
        help="Manifest column containing tracking CSV paths.",
    )
    parser.add_argument(
        "--frame-column",
        default="Frame",
        help="Tracking CSV column containing frame numbers.",
    )
    parser.add_argument(
        "--x-column",
        default="X",
        help="Tracking CSV column containing X coordinates.",
    )
    parser.add_argument(
        "--y-column",
        default="Y",
        help="Tracking CSV column containing Y coordinates.",
    )
    parser.add_argument(
        "--num-frames-per-video",
        type=int,
        default=200,
        help="Number of evenly spaced labeled frames to export per video.",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=None,
        help="Use every Nth valid tracking row instead of even spacing.",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Export every valid frame in each tracking CSV.",
    )
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=0,
        help="Subtract this value from tracking frame numbers before extraction.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.95,
        help="Training fraction to write into the DLC config.",
    )
    parser.add_argument(
        "--dotsize",
        type=int,
        default=6,
        help="Point size to write into the DLC config.",
    )
    parser.add_argument(
        "--image-format",
        choices=("png", "jpg"),
        default="png",
        help="Image format for extracted training frames.",
    )
    parser.add_argument(
        "--copy-videos",
        action="store_true",
        help="Copy videos into the DLC project instead of referencing them in place.",
    )
    parser.add_argument(
        "--create-training-dataset",
        action="store_true",
        help="Also run deeplabcut.create_training_dataset at the end.",
    )
    return parser.parse_args()


def require_deeplabcut():
    try:
        import deeplabcut
    except ImportError as exc:
        raise SystemExit(
            "deeplabcut is not importable in this Python environment. "
            "Run this script from your DeepLabCut conda environment."
        ) from exc
    return deeplabcut


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "opencv-python (cv2) is not available in this Python environment. "
            "Run this script from an environment that can read and write video frames."
        ) from exc
    return cv2


def resolve_manifest_path(raw_path: str, manifest_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (manifest_dir / path).resolve()
    return path


def load_manifest(args: argparse.Namespace) -> list[PairRecord]:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest_dir = manifest_path.parent
    manifest = pd.read_csv(manifest_path)

    missing_columns = {
        args.manifest_video_column,
        args.manifest_csv_column,
    } - set(manifest.columns)
    if missing_columns:
        raise SystemExit(
            f"Manifest is missing required columns: {sorted(missing_columns)}"
        )

    pairs: list[PairRecord] = []
    for row in manifest.to_dict(orient="records"):
        video_path = resolve_manifest_path(
            str(row[args.manifest_video_column]), manifest_dir
        )
        csv_path = resolve_manifest_path(
            str(row[args.manifest_csv_column]), manifest_dir
        )
        if not video_path.exists():
            raise SystemExit(f"Video file does not exist: {video_path}")
        if not csv_path.exists():
            raise SystemExit(f"Tracking CSV does not exist: {csv_path}")
        pairs.append(PairRecord(video_path=video_path, csv_path=csv_path))

    clip_names = [pair.clip_name for pair in pairs]
    duplicate_names = sorted({name for name in clip_names if clip_names.count(name) > 1})
    if duplicate_names:
        raise SystemExit(
            "Video basenames must be unique because DLC labeled-data folders are "
            f"derived from them. Duplicate names: {duplicate_names}"
        )

    return pairs


def create_project(
    deeplabcut,
    args: argparse.Namespace,
    pairs: list[PairRecord],
) -> Path:
    working_directory = Path(args.working_directory).expanduser().resolve()
    working_directory.mkdir(parents=True, exist_ok=True)

    config_path = deeplabcut.create_new_project(
        args.project_name,
        args.experimenter,
        [str(pair.video_path) for pair in pairs],
        working_directory=str(working_directory),
        copy_videos=args.copy_videos,
        multianimal=False,
    )
    return Path(config_path).resolve()


def update_config(config_path: Path, args: argparse.Namespace) -> None:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["scorer"] = args.experimenter
    config["bodyparts"] = [args.bodypart]
    config["skeleton"] = []
    config["dotsize"] = args.dotsize
    config["TrainingFraction"] = [args.train_fraction]

    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def get_sample_positions(total_rows: int, requested_rows: int) -> list[int]:
    if requested_rows <= 0:
        raise SystemExit("--num-frames-per-video must be greater than zero.")
    if total_rows <= requested_rows:
        return list(range(total_rows))
    if requested_rows == 1:
        return [0]
    denominator = requested_rows - 1
    max_index = total_rows - 1
    return sorted(
        {
            round(index * max_index / denominator)
            for index in range(requested_rows)
        }
    )


def load_tracking_rows(args: argparse.Namespace, csv_path: Path) -> pd.DataFrame:
    tracking = pd.read_csv(csv_path)
    missing_columns = {
        args.frame_column,
        args.x_column,
        args.y_column,
    } - set(tracking.columns)
    if missing_columns:
        raise SystemExit(
            f"Tracking CSV {csv_path} is missing columns: {sorted(missing_columns)}"
        )

    tracking = tracking[[args.frame_column, args.x_column, args.y_column]].copy()
    tracking = tracking.rename(
        columns={
            args.frame_column: "frame",
            args.x_column: "x",
            args.y_column: "y",
        }
    )
    tracking = tracking.dropna(subset=["frame", "x", "y"])
    tracking["frame"] = tracking["frame"].astype(int) - args.frame_offset
    tracking = tracking[tracking["frame"] >= 0]
    tracking = tracking.sort_values("frame").drop_duplicates(subset="frame", keep="first")
    return tracking.reset_index(drop=True)


def sample_tracking_rows(args: argparse.Namespace, tracking: pd.DataFrame) -> pd.DataFrame:
    if tracking.empty:
        return tracking

    if args.all_frames:
        return tracking.copy()

    if args.every_n is not None:
        if args.every_n <= 0:
            raise SystemExit("--every-n must be greater than zero.")
        return tracking.iloc[:: args.every_n].reset_index(drop=True)

    positions = get_sample_positions(len(tracking), args.num_frames_per_video)
    return tracking.iloc[positions].reset_index(drop=True)


def extract_frames_and_write_labels(
    cv2,
    project_dir: Path,
    scorer: str,
    bodypart: str,
    image_format: str,
    pair: PairRecord,
    tracking_rows: pd.DataFrame,
) -> int:
    labeled_dir = project_dir / "labeled-data" / pair.clip_name
    labeled_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(pair.video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {pair.video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    valid_rows = tracking_rows[tracking_rows["frame"] < frame_count].copy()
    if valid_rows.empty:
        cap.release()
        return 0

    labels: list[tuple[str, float, float]] = []
    for row in valid_rows.itertuples(index=False):
        frame_number = int(row.frame)
        image_name = f"img{frame_number:08d}.{image_format}"
        relative_image_path = Path("labeled-data") / pair.clip_name / image_name
        output_image_path = labeled_dir / image_name

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, image = cap.read()
        if not success:
            print(
                f"Warning: could not extract frame {frame_number} from {pair.video_path}",
                file=sys.stderr,
            )
            continue

        if not cv2.imwrite(str(output_image_path), image):
            raise SystemExit(f"Could not write extracted frame to {output_image_path}")

        labels.append((relative_image_path.as_posix(), float(row.x), float(row.y)))

    cap.release()

    if not labels:
        return 0

    columns = pd.MultiIndex.from_product(
        [[scorer], [bodypart], ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    label_index = [item[0] for item in labels]
    label_values = [[item[1], item[2]] for item in labels]
    label_frame = pd.DataFrame(label_values, index=label_index, columns=columns)

    csv_output = labeled_dir / f"CollectedData_{scorer}.csv"
    label_frame.to_csv(csv_output)
    return len(labels)


def build_labels_for_pairs(
    cv2,
    args: argparse.Namespace,
    config_path: Path,
    pairs: Iterable[PairRecord],
) -> int:
    project_dir = config_path.parent
    total_labels = 0

    for pair in pairs:
        print(f"Processing {pair.video_path.name}")
        tracking = load_tracking_rows(args, pair.csv_path)
        sampled_tracking = sample_tracking_rows(args, tracking)
        labels_written = extract_frames_and_write_labels(
            cv2=cv2,
            project_dir=project_dir,
            scorer=args.experimenter,
            bodypart=args.bodypart,
            image_format=args.image_format,
            pair=pair,
            tracking_rows=sampled_tracking,
        )
        print(f"  wrote {labels_written} labeled frames")
        total_labels += labels_written

    return total_labels


def main() -> None:
    args = parse_args()
    deeplabcut = require_deeplabcut()
    cv2 = require_cv2()

    pairs = load_manifest(args)
    config_path = create_project(deeplabcut, args, pairs)
    update_config(config_path, args)

    total_labels = build_labels_for_pairs(cv2, args, config_path, pairs)
    deeplabcut.convertcsv2h5(str(config_path))

    if args.create_training_dataset:
        deeplabcut.create_training_dataset(str(config_path))

    print("")
    print(f"DeepLabCut config created at: {config_path}")
    print(f"Total labeled frames written: {total_labels}")
    if not args.create_training_dataset:
        print("Next steps:")
        print(f"  1. Optionally inspect labels with: deeplabcut.check_labels(r'{config_path}')")
        print(f"  2. Create a training dataset with: deeplabcut.create_training_dataset(r'{config_path}')")
        print(f"  3. Train with: deeplabcut.train_network(r'{config_path}')")


if __name__ == "__main__":
    main()
