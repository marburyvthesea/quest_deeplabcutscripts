#!/usr/bin/env python3
"""dlc_superanimal_topview.py — SuperAnimal-TopViewMouse inference for T-maze videos.

Replaces the `create_pretrained_project(..., model="superanimal_topviewmouse")`
route in quest_deeplabcutscripts, which goes through the TensorFlow DLCRNet
weights. The PyTorch path below is the one that supports video adaptation and
the HRNet-w32 top-down pose model, and it is where DeepLabCut development has
moved.

Signature reference (deeplabcut/modelzoo/video_inference.py, main branch):
    video_inference_superanimal(videos, superanimal_name, model_name,
        detector_name=None, scale_list=None, video_extensions=None,
        dest_folder=None, cropping=None, video_adapt=False, batch_size=1,
        detector_batch_size=1, pcutoff=0.1, adapt_iterations=1000,
        pseudo_threshold=0.1, bbox_threshold=0.9, detector_epochs=4,
        pose_epochs=4, max_individuals=10, video_adapt_batch_size=8, ...)

Engine routing is by model_name: `dlcrnet` -> TensorFlow, everything else ->
PyTorch. Older DLC builds name the extension argument `videotype` rather than
`video_extensions`; this script probes the signature and adapts.

Usage
-----
  python dlc_superanimal_topview.py VIDEO_OR_DIR [-o OUTDIR] [--ext .mp4]
         [--adapt] [--pcutoff 0.15] [--batch-size 8] [--max-individuals 1]
         [--cropping x1 x2 y1 y2] [--limit N]

Run with --adapt on ONE representative video per rig/animal first; the adapted
checkpoint generalizes to visually similar sessions, so the remaining videos can
be analyzed without repeating adaptation.
"""
from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

os.environ.setdefault("DLClight", "True")

POSE_MODEL = "hrnet_w32"                              # or "resnet_50"
DETECTOR = "fasterrcnn_mobilenet_v3_large_fpn"        # or "fasterrcnn_resnet50_fpn_v2"
SUPERANIMAL = "superanimal_topviewmouse"              # 27 keypoints, top-down mouse


def collect(target: Path, ext: str, limit: int | None) -> list[str]:
    vids = sorted(target.rglob(f"*{ext}")) if target.is_dir() else [target]
    if limit:
        vids = vids[:limit]
    if not vids:
        raise SystemExit(f"no {ext} videos under {target}")
    return [str(v) for v in vids]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos")
    p.add_argument("-o", "--out", default=None, help="dest_folder for h5/csv output")
    p.add_argument("--ext", default=".mp4")
    p.add_argument("--adapt", action="store_true", help="run video adaptation (self-training)")
    p.add_argument("--pcutoff", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--detector-batch-size", type=int, default=4)
    p.add_argument("--max-individuals", type=int, default=1)
    p.add_argument("--pose-model", default=POSE_MODEL)
    p.add_argument("--detector", default=DETECTOR)
    p.add_argument("--cropping", nargs=4, type=int, default=None,
                   metavar=("X1", "X2", "Y1", "Y2"),
                   help="crop applied to ALL videos; prefer pre-cropping with maze_prep.py")
    p.add_argument("--no-labeled-video", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()

    import deeplabcut

    vids = collect(Path(a.videos), a.ext, a.limit)
    print(f"deeplabcut {deeplabcut.__version__}; {len(vids)} video(s)")
    for v in vids:
        print("  ", v)

    fn = deeplabcut.video_inference_superanimal
    params = set(inspect.signature(fn).parameters)

    kw = dict(
        superanimal_name=SUPERANIMAL,
        model_name=a.pose_model,
        detector_name=a.detector,
        video_adapt=a.adapt,
        pcutoff=a.pcutoff,
        batch_size=a.batch_size,
        detector_batch_size=a.detector_batch_size,
        max_individuals=a.max_individuals,
    )
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        kw["dest_folder"] = a.out
    if a.cropping:
        kw["cropping"] = list(a.cropping)
    if a.no_labeled_video and "create_labeled_video" in params:
        kw["create_labeled_video"] = False
    # extension argument was renamed between releases
    if "video_extensions" in params:
        kw["video_extensions"] = a.ext
    elif "videotype" in params:
        kw["videotype"] = a.ext
    kw = {k: v for k, v in kw.items() if k in params}

    print("call kwargs:", {k: v for k, v in kw.items() if k != "videos"})
    fn(vids, **kw)
    print("inference finished")


if __name__ == "__main__":
    main()
