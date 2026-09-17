#!/usr/bin/env python
"""
Check (and optionally fetch) the SuperAnimal snapshots the fine-tune needs.

DLC 3.0.1 has no model-zoo cache environment variable: `get_snapshot_folder_path()`
is hard-wired to <deeplabcut package>/modelzoo/checkpoints. So `DLC_MODELZOO_CACHE`
does nothing, and the snapshots live inside the conda env whether you like it or
not. Don't guess the filenames -- ask the API, which is what this does.

`build_weight_init` will download a missing snapshot itself, but it does that
*inside* the training job. If the compute node has no outbound network that
fails after the GPU is allocated, so check first on a login node:

  python dlc_snapshots.py              # report only; exit 1 if any are missing
  python dlc_snapshots.py --download   # fetch the missing ones (needs network)
"""
import argparse
import sys

SUPER_ANIMAL = "superanimal_topviewmouse"
MODELS = ("hrnet_w32", "fasterrcnn_mobilenet_v3_large_fpn")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download", action="store_true",
                   help="fetch missing snapshots (run on a login node)")
    p.add_argument("--models", nargs="+", default=list(MODELS))
    a = p.parse_args()

    from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
        download_super_animal_snapshot,
        get_snapshot_folder_path,
        get_super_animal_snapshot_path,
    )

    folder = get_snapshot_folder_path()
    print(f"snapshot dir: {folder}")
    if folder.exists():
        names = sorted(q.name for q in folder.iterdir())
        print(f"contents ({len(names)}): {names if names else '<empty>'}")
    else:
        print("contents: <folder does not exist>")

    missing = []
    for model in a.models:
        path = get_super_animal_snapshot_path(SUPER_ANIMAL, model, download=False)
        if path.exists():
            size = path.stat().st_size / 1e6
            print(f"  found   {path.name}  ({size:.0f} MB)")
        else:
            print(f"  MISSING {path.name}")
            missing.append(model)

    if not missing:
        print("all snapshots present; the training job needs no network")
        return

    if not a.download:
        print(f"\n{len(missing)} snapshot(s) missing. Fetch them on a login node:\n"
              f"  python dlc_snapshots.py --download\n"
              f"Do not let the training job download them: a compute node may have "
              f"no outbound network, and the failure would land after the GPU is "
              f"allocated.")
        sys.exit(1)

    for model in missing:
        print(f"downloading {SUPER_ANIMAL}_{model} ...", flush=True)
        out = download_super_animal_snapshot(SUPER_ANIMAL, model)
        print(f"  -> {out}", flush=True)

    still = [m for m in a.models
             if not get_super_animal_snapshot_path(SUPER_ANIMAL, m,
                                                   download=False).exists()]
    if still:
        print(f"still missing after download: {still}")
        sys.exit(1)
    print("all snapshots present")


if __name__ == "__main__":
    main()
