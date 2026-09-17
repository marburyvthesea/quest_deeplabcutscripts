#!/usr/bin/env python
"""
Fine-tune SuperAnimal-TopViewMouse on blob-tracker labels.  Pinned to DLC 3.0.1.

Why this works despite supervising only one keypoint
----------------------------------------------------
DLC derives the detector's training boxes from the labelled keypoints
(`bbox_from_keypoints`: min/max over visible keypoints, +- `data.bbox_margin`,
default 20). With one labelled keypoint that is a fixed 40x40 box centred on the
body centre -- which for a mouse at fixed camera height is a reasonable box, so
the detector gets a sane target rather than a degenerate one.

`memory_replay` then pseudo-labels the other 26 keypoints, and in 3.0.1 it does
so *inside the ground-truth boxes* (the detector call in
`pose_estimation_pytorch/modelzoo/memory_replay.py` is commented out; pose
inputs are built from the annotation bboxes). So the pretrained pose model sees
a tight crop centred on the real animal, not the whole frame where it locks onto
the cable tangle in the maze voids. That is why the pseudo-labels here should be
usable, and why the pretrained posture head may survive rather than merely be
frozen. Treat the resulting non-centre keypoints as pseudo-labels, not ground
truth: they are good enough to keep the head alive, not to make posture claims
without a hand-labelled check.

`with_decoder=True` is mandatory for memory replay, and it requires a conversion
table in the project config. This script creates it, and it is the identity map
only because the project's bodyparts were read from your own SuperAnimal
prediction .h5 -- verified to match `superanimal_topviewmouse.yaml` in both
membership and order, with mouse_center at index 9. Rename a keypoint and this
assumption breaks.

Detector defaults to fasterrcnn_mobilenet_v3_large_fpn to match the snapshots
already in your model-zoo cache from the inference run, so the job does not need
HuggingFace access from a compute node.

Run dlc_probe.py first.

  python dlc_finetune.py --config .../config.yaml --epochs 100
"""
import argparse
import inspect
import os
import sys

SUPER_ANIMAL = "superanimal_topviewmouse"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--net-type", default="top_down_hrnet_w32")
    p.add_argument("--model-name", default="hrnet_w32",
                   help="SuperAnimal pose architecture whose weights to load")
    p.add_argument("--detector-name", default="fasterrcnn_mobilenet_v3_large_fpn",
                   help="matches the snapshots already in your model-zoo cache")
    p.add_argument("--detector-epochs", type=int, default=100,
                   help="the detector is what actually failed zero-shot; train it")
    p.add_argument("--shuffle", type=int, default=1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--save-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--no-memory-replay", action="store_true")
    p.add_argument("--skip-dataset", action="store_true",
                   help="training dataset already created")
    p.add_argument("--analyze", default=None, help="video to analyze after training")
    p.add_argument("--dest", default=None, help="where to write analysis output")
    a = p.parse_args()

    import deeplabcut
    print(f"deeplabcut {deeplabcut.__version__}", flush=True)
    cfg = os.path.abspath(a.config)

    if not a.skip_dataset:
        from deeplabcut.modelzoo.utils import create_conversion_table
        # 3.0.1 moved this: auxiliaryfunctions.read_config is gone, and
        # core.config.read_config returns a pydantic ProjectConfig, not a dict.
        try:
            from deeplabcut.core.config import read_config_as_dict as _read
        except ImportError:
            from deeplabcut.utils.auxiliaryfunctions import read_plainconfig as _read
        c = _read(cfg)
        bp = list(c["bodyparts"])

        # Identity conversion table. with_decoder=True is refused by
        # WeightInitialization without one, and memory_replay is refused without
        # with_decoder -- so this is not optional, it is the prerequisite.
        table = {b: b for b in bp}
        ct = create_conversion_table(cfg, SUPER_ANIMAL, table)
        print(f"conversion table written for {len(table)} bodyparts; "
              f"array={list(ct.to_array())}", flush=True)

        try:
            from deeplabcut.modelzoo import build_weight_init
        except ImportError:                       # pre-3.0 layout
            from deeplabcut.core.weight_init import WeightInitialization
            build_weight_init = WeightInitialization.build
        # pass the PATH, not the dict read above: create_conversion_table has
        # just written SuperAnimalConversionTables into the file, and
        # build_weight_init needs to see it.
        wi = build_weight_init(
            cfg=cfg,
            super_animal=SUPER_ANIMAL,
            model_name=a.model_name,
            detector_name=a.detector_name,
            with_decoder=True,
            memory_replay=not a.no_memory_replay,
        )
        print(f"weight init: pose={wi.snapshot_path} detector={wi.detector_snapshot_path}\n"
              f"  with_decoder={wi.with_decoder} memory_replay={wi.memory_replay}", flush=True)
        if wi.snapshot_path is None:
            print("no SuperAnimal pose snapshot resolved -- training would start from "
                  "ImageNet weights, which with one labelled keypoint learns nothing. "
                  "Check <deeplabcut>/modelzoo/checkpoints and rerun.", flush=True)
            sys.exit(2)
        # train_network raises if memory_replay is on and this is None -- and it
        # raises *after* the GPU is allocated. The requirement is vestigial (the
        # detector inference in memory_replay.py is commented out; pseudo-labels
        # use the ground-truth boxes) but it is enforced, so fail here instead.
        if wi.memory_replay and wi.detector_snapshot_path is None:
            print(f"memory replay needs a detector snapshot and none resolved for "
                  f"'{a.detector_name}'. Either fetch it on a login node or rerun "
                  f"with --no-memory-replay.", flush=True)
            sys.exit(2)

        deeplabcut.create_training_dataset(
            cfg, net_type=a.net_type, detector_type=a.detector_name, weight_init=wi)
        print("training dataset created", flush=True)

    tkw = dict(shuffle=a.shuffle)
    tsig = inspect.signature(deeplabcut.train_network).parameters
    for k, v in (("epochs", a.epochs), ("save_epochs", a.save_epochs),
                 ("batch_size", a.batch_size), ("display_iters", 100),
                 ("detector_epochs", a.detector_epochs),
                 ("detector_batch_size", a.batch_size)):
        if k in tsig:
            tkw[k] = v
    print(f"train_network({tkw})", flush=True)
    deeplabcut.train_network(cfg, **tkw)

    # 3.0.1 spells this `shuffles` (lower case); 2.x used `Shuffles`.
    esig = inspect.signature(deeplabcut.evaluate_network).parameters
    ekw = {("shuffles" if "shuffles" in esig else "Shuffles"): [a.shuffle]}
    try:
        deeplabcut.evaluate_network(cfg, plotting=True, per_keypoint_evaluation=True,
                                    **ekw)
    except Exception as e:
        print(f"evaluate_network failed ({type(e).__name__}: {e})", flush=True)

    if a.analyze:
        dest = a.dest or os.path.dirname(a.analyze)
        deeplabcut.analyze_videos(cfg, [a.analyze], shuffle=a.shuffle,
                                  destfolder=dest, save_as_csv=True)
        try:
            deeplabcut.create_labeled_video(cfg, [a.analyze], shuffle=a.shuffle,
                                            destfolder=dest)
        except Exception as e:
            print(f"create_labeled_video failed ({type(e).__name__})", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
