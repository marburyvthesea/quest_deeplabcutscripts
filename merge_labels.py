"""Merge updated CollectedData label files from a transferred project copy.

Why merge rather than adopt the new copy: the transferred project carries the
updated labels but not the trained model (no dlc-models-pytorch), not the full
set of extracted images (50 per animal instead of 250 for C/F/K), and its
config.yaml still has project_path pointing at the Mac mount it was labelled
on, which DLC resolves labeled-data against. The destination project has all
three correct. Only the label files changed, so only they need to move.

Safety: nothing is deleted or moved. Destination label files are copied into a
timestamped backup directory inside the project before being overwritten, and
the merge refuses outright if the two sides are not structurally compatible
(different bodypart columns, different scorer, or a disjoint image index),
because a silent mismatch there would corrupt the training set.

Images referenced by new rows but absent from the destination are copied too,
so the merged CollectedData never points at a missing file.

  python merge_labels.py --src ... --dst ...            # dry run, default
  python merge_labels.py --src ... --dst ... --apply    # perform the merge
"""

import argparse
import glob
import os
import shutil
import time

import numpy as np
import pandas as pd

SRC = "/projects/b1118/dlc_analysis/updatedProject_2/TmazeParts-jjm-2026-10-20"
DST = "/projects/b1118/dlc_analysis/TmazeParts-jjm-2026-10-20"


def counts(df):
    """rows, fully-labelled, partially-labelled, blank."""
    A = df.to_numpy(float)
    xy = A.reshape(len(A), -1, 2)
    vis = np.isfinite(xy).all(2)
    n, k = vis.sum(1), vis.shape[1]
    return len(df), int((n == k).sum()), int(((n > 0) & (n < k)).sum()), int((n == 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC, help="project copy holding the new labels")
    ap.add_argument("--dst", default=DST, help="project to merge into (keeps its model)")
    ap.add_argument("--apply", action="store_true", help="actually copy; default is a dry run")
    a = ap.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = os.path.join(a.dst, f"labelbackup-{stamp}")

    plan, problems = [], []
    for d in sorted(glob.glob(os.path.join(a.src, "labeled-data", "*"))):
        if not os.path.isdir(d) or d.endswith("_labeled"):
            continue
        v = os.path.basename(d)
        s_h5 = sorted(glob.glob(os.path.join(d, "CollectedData_*.h5")))
        if not s_h5:
            continue
        ddir = os.path.join(a.dst, "labeled-data", v)
        d_h5 = sorted(glob.glob(os.path.join(ddir, "CollectedData_*.h5")))

        S = pd.read_hdf(s_h5[0])
        if not d_h5:
            problems.append(f"{v}: destination has no CollectedData -- new directory, "
                            "copy it wholesale by hand if that is intended")
            continue
        if not os.path.isdir(ddir):
            problems.append(f"{v}: destination directory missing")
            continue
        D = pd.read_hdf(d_h5[0])

        if list(S.columns) != list(D.columns):
            problems.append(f"{v}: COLUMNS DIFFER -- refusing "
                            f"(src {len(S.columns)} cols, dst {len(D.columns)})")
            continue
        si, di = set(map(str, S.index)), set(map(str, D.index))
        if not (si & di):
            problems.append(f"{v}: index formats are disjoint -- refusing")
            continue

        # Images a new row points at that the destination does not have.
        missing = []
        for ix in S.index:
            fn = ix[-1] if isinstance(ix, tuple) else os.path.basename(str(ix))
            if not os.path.exists(os.path.join(ddir, fn)):
                src_img = os.path.join(d, fn)
                missing.append((src_img, os.path.join(ddir, fn), os.path.exists(src_img)))

        plan.append(dict(video=v, src_h5=s_h5[0], dst_h5=d_h5[0],
                         src_csv=s_h5[0][:-3] + ".csv", dst_csv=d_h5[0][:-3] + ".csv",
                         before=counts(D), after=counts(S),
                         new_rows=len(si - di), missing=missing))

    print(f"{'video':16s} {'rows':>12s} {'full':>12s} {'part':>12s} {'blank':>12s}  new  img")
    tb = np.zeros(4, int); ta = np.zeros(4, int)
    for p in plan:
        b, af = p["before"], p["after"]
        tb += np.array(b); ta += np.array(af)
        print(f"{p['video']:16s} "
              + " ".join(f"{b[i]:5d}->{af[i]:5d}" for i in range(4))
              + f"  {p['new_rows']:3d}  {len(p['missing']):3d}")
    print(f"{'TOTAL':16s} " + " ".join(f"{tb[i]:5d}->{ta[i]:5d}" for i in range(4)))
    print(f"\ncomplete labels: {tb[1]} -> {ta[1]}  (+{ta[1]-tb[1]})")
    print(f"partial labels:  {tb[2]} -> {ta[2]}  (+{ta[2]-tb[2]})")
    print(f"blank rows:      {tb[3]} -> {ta[3]}  ({ta[3]-tb[3]:+d})")

    if problems:
        print("\n!! PROBLEMS -- these directories will NOT be merged:")
        for p in problems:
            print("   " + p)

    nmiss = sum(len(p["missing"]) for p in plan)
    unavail = [m for p in plan for m in p["missing"] if not m[2]]
    if nmiss:
        print(f"\n{nmiss} referenced image(s) absent from destination; "
              f"{len(unavail)} of those are also absent from the source")
        for m in unavail:
            print("   !! cannot supply:", m[1])

    if not a.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply to perform the merge.")
        return

    if unavail:
        raise SystemExit("refusing to apply: a merged row would point at a missing image")

    os.makedirs(backup, exist_ok=True)
    n_lab = n_img = 0
    for p in plan:
        bdir = os.path.join(backup, p["video"])
        os.makedirs(bdir, exist_ok=True)
        for dst in (p["dst_h5"], p["dst_csv"]):
            if os.path.exists(dst):
                shutil.copy2(dst, os.path.join(bdir, os.path.basename(dst)))
        for src, dst in ((p["src_h5"], p["dst_h5"]), (p["src_csv"], p["dst_csv"])):
            if os.path.exists(src):
                shutil.copy2(src, dst); n_lab += 1
        for src_img, dst_img, ok in p["missing"]:
            shutil.copy2(src_img, dst_img); n_img += 1

    print(f"\ncopied {n_lab} label file(s) and {n_img} image(s)")
    print(f"previous label files preserved in {backup}")
    print("\nNext: bump `iteration: 0` -> `iteration: 1` in the destination config.yaml "
          "before create_training_dataset, so the Oct-20 model under iteration-0 "
          "stays intact and the existing benchmark remains reproducible.")


if __name__ == "__main__":
    main()
