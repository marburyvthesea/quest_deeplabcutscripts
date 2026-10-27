#!/usr/bin/env python3
"""Extract a fixed number of DLC k-means frames from selected project videos."""

import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--video-stems", nargs="+", required=True)
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--cluster-step", type=int, default=10)
    parser.add_argument("--cluster-resizewidth", type=int, default=30)
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="allow extraction into folders that already contain img*.png",
    )
    return parser.parse_args()


def select_video_sets(cfg, wanted):
    video_sets = cfg.get("video_sets_original") or cfg.get("video_sets", {})
    matches = {}
    duplicate_stems = set()
    for video, settings in video_sets.items():
        stem = Path(video).stem
        if stem not in wanted:
            continue
        if stem in matches:
            duplicate_stems.add(stem)
        matches[stem] = (video, settings)

    if duplicate_stems:
        names = ", ".join(sorted(duplicate_stems))
        raise SystemExit(f"multiple configured videos have these stems: {names}")

    missing = wanted - matches.keys()
    if missing:
        available = ", ".join(sorted(Path(v).stem for v in video_sets))
        raise SystemExit(
            f"video stems not present in config: {', '.join(sorted(missing))}\n"
            f"available stems: {available}"
        )

    return {matches[stem][0]: matches[stem][1] for stem in sorted(wanted)}


def main():
    args = parse_args()
    if args.num_frames < 1:
        raise SystemExit("--num-frames must be at least 1")
    if args.cluster_step < 1:
        raise SystemExit("--cluster-step must be at least 1")

    os.environ.setdefault("DLClight", "True")
    os.environ.setdefault("MPLBACKEND", "Agg")

    import deeplabcut
    from deeplabcut.utils import auxiliaryfunctions

    config = args.config.expanduser().resolve()
    if not config.is_file():
        raise SystemExit(f"config not found: {config}")

    wanted = set(args.video_stems)
    cfg = auxiliaryfunctions.read_config(str(config))
    selected = select_video_sets(cfg, wanted)

    for video in selected:
        if not Path(video).is_file():
            raise SystemExit(f"configured video not found: {video}")
        output = config.parent / "labeled-data" / Path(video).stem
        existing = sorted(output.glob("img*.png")) if output.exists() else []
        if existing and not args.allow_existing:
            raise SystemExit(
                f"{output} already contains {len(existing)} extracted frames; "
                "remove them or pass --allow-existing"
            )

    # DLC reads numframes2pick and the video list from disk. A temporary config
    # in the project directory limits extraction to these videos without changing
    # the real config.yaml or revisiting videos that have already been labeled.
    cfg["video_sets"] = selected
    cfg.pop("video_sets_original", None)
    cfg["numframes2pick"] = args.num_frames
    temporary_config = config.parent / f".extract_kmeans_{os.getpid()}.yaml"
    auxiliaryfunctions.write_config(str(temporary_config), cfg)

    print("Selected videos:")
    for video in selected:
        print(f"  {video}")
    print(
        f"Extracting {args.num_frames} frames per video with k-means "
        f"(cluster_step={args.cluster_step}, "
        f"resizewidth={args.cluster_resizewidth})"
    )

    try:
        deeplabcut.extract_frames(
            str(temporary_config),
            mode="automatic",
            algo="kmeans",
            crop=False,
            userfeedback=False,
            cluster_step=args.cluster_step,
            cluster_resizewidth=args.cluster_resizewidth,
            cluster_color=False,
            opencv=True,
        )
    finally:
        temporary_config.unlink(missing_ok=True)

    for video in selected:
        output = config.parent / "labeled-data" / Path(video).stem
        count = len(list(output.glob("img*.png")))
        print(f"{Path(video).stem}: {count} extracted frame(s) in {output}")
        if count < args.num_frames:
            raise SystemExit(
                f"expected {args.num_frames} frames for {Path(video).stem}, found {count}"
            )


if __name__ == "__main__":
    main()
