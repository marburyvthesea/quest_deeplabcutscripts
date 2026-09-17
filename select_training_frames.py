#!/usr/bin/env python
"""
Pick training frames from blob-tracker output, gated hard and spread widely.

The point of the gate is that these coordinates become training labels: a
detection that is merely probable is worse than no label at all, because the
network will learn it. So this keeps only frames where the blob is dark,
compact, the right size, moving, not interpolated, not on a furniture site, and
flanked by detections on both sides. Frames are then chosen to spread over
position and time rather than taken at a fixed stride, since a stride
oversamples wherever the animal spent its time.

Images must come from the RAW video. Extracting from a DLC-labeled overlay
would bake the old (wrong) markers into the training images.

  python select_training_frames.py --track C_blobtrack_qc.csv --video C.avi \
         --geom maze_geometry_C.json --n 200 --outdir C_trainframes \
         --manifest C_trainframes_manifest.csv --fig C_trainframes.png
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd


def gate(T, a):
    f = np.isfinite(T.x) & np.isfinite(T.y)
    reasons = {}
    def add(name, cond):
        reasons[name] = int((f & ~cond).sum())
        return cond
    f &= add("not usable", T.get("usable", pd.Series(True, index=T.index)).fillna(False))
    f &= add("interpolated", ~T.get("interp", pd.Series(False, index=T.index)).fillna(False))
    f &= add("stationary", ~T.get("stationary", pd.Series(False, index=T.index)).fillna(False))
    f &= add("too shallow", T.depth >= a.min_depth)
    f &= add("not compact", T.extent >= a.min_extent)
    f &= add("too elongated", T.elong <= a.max_elong)
    f &= add("area out of range", (T.area >= a.area[0]) & (T.area <= a.area[1]))
    f &= add("implausible speed", T.speed_px <= a.max_speed)
    nb = np.isfinite(T.x.shift(1)) & np.isfinite(T.x.shift(-1))
    f &= add("isolated detection", nb.fillna(False))
    return f, reasons


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", required=True, help="output of blob_tracker.py post")
    p.add_argument("--video", required=True, help="RAW video, not a labeled overlay")
    p.add_argument("--geom", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--outdir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--fig", default=None)
    p.add_argument("--min-depth", type=float, default=80)
    p.add_argument("--min-extent", type=float, default=0.40)
    p.add_argument("--max-elong", type=float, default=3.0)
    p.add_argument("--area", type=int, nargs=2, default=[350, 1400])
    p.add_argument("--max-speed", type=float, default=15.0)
    p.add_argument("--time-weight", type=float, default=0.5,
                   help="weight on frame index relative to position when spreading")
    p.add_argument("--allow-labeled", action="store_true")
    p.add_argument("--no-images", action="store_true",
                   help="write the manifest only; extract images later on the machine "
                        "that holds the raw video")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    if ("labeled" in os.path.basename(a.video).lower()
            and not a.allow_labeled and not a.no_images):
        raise SystemExit(
            f"refusing to extract training images from {os.path.basename(a.video)}: "
            "the filename says it is a DLC overlay, which would bake the old markers "
            "into the training set. Point --video at the raw AVI (or pass "
            "--allow-labeled if you are certain)."
        )

    T = pd.read_csv(a.track)
    f, reasons = gate(T, a)
    G = T.loc[f]
    print(f"{len(T)} frames -> {len(G)} pass the gate ({100*len(G)/len(T):.2f}%)")
    print("rejected by:")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {v:8d}  {k}")
    if len(G) < a.n:
        print(f"WARNING: only {len(G)} gated frames, fewer than the {a.n} requested")

    # spread over position and time
    from sklearn.cluster import KMeans
    X = np.column_stack([
        (G.x - G.x.mean()) / max(G.x.std(), 1e-6),
        (G.y - G.y.mean()) / max(G.y.std(), 1e-6),
        a.time_weight * (G.frame - G.frame.mean()) / max(G.frame.std(), 1e-6),
    ])
    k = min(a.n, len(G))
    km = KMeans(n_clusters=k, n_init=4, random_state=a.seed).fit(X)
    pick = []
    for c in range(k):
        idx = np.flatnonzero(km.labels_ == c)
        if idx.size:
            d = np.linalg.norm(X[idx] - km.cluster_centers_[c], axis=1)
            pick.append(G.index[idx[int(np.argmin(d))]])
    S = T.loc[sorted(pick)].copy()
    print(f"\nselected {len(S)} frames; "
          f"frame span {int(S.frame.min())}-{int(S.frame.max())}, "
          f"median gap {int(np.median(np.diff(S.frame.to_numpy())))} frames")

    if a.no_images:
        S["image"] = [f"img{int(f):07d}.png" for f in S.frame]
        S[["frame", "image", "x", "y", "area", "depth", "extent", "elong", "score",
           "speed_px"]].to_csv(a.manifest, index=False)
        print(f"wrote {a.manifest} ({len(S)} rows, no images extracted)")
    else:
        os.makedirs(a.outdir, exist_ok=True)
    cap = None
    if not a.no_images:
        cap = cv2.VideoCapture(a.video)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {a.video}")
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_video != len(T):
            print(f"WARNING: video has {n_video} frames, track table has {len(T)} -- "
                  "indices may not correspond")
    names = []
    for fr in ([] if a.no_images else S.frame.astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
        ok, im = cap.read()
        if not ok:
            names.append("")
            continue
        nm = f"img{int(fr):07d}.png"
        cv2.imwrite(os.path.join(a.outdir, nm), im)
        names.append(nm)
    if cap is not None:
        cap.release()
    if not a.no_images:
        S["image"] = names
        S = S.loc[S.image != ""]
        S[["frame", "image", "x", "y", "area", "depth", "extent", "elong", "score",
           "speed_px"]].to_csv(a.manifest, index=False)
        print(f"wrote {len(S)} images to {a.outdir}/ and {a.manifest}")

    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        g = json.load(open(a.geom))
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        ok = np.isfinite(T.x)
        ax[0].scatter(T.x[ok], T.y[ok], s=1, c="#cfd8dc", lw=0, label="all detections")
        ax[0].scatter(S.x, S.y, s=16, c="#c62828", lw=0, label="selected")
        b = g["maze_outer"]
        ax[0].add_patch(plt.Rectangle((b["x"], b["y"]), b["w"], b["h"], fill=False, ec="k", lw=1))
        for b in g.get("exclude_islands", []):
            ax[0].add_patch(plt.Rectangle((b["x"], b["y"]), b["w"], b["h"], fill=False,
                                          ec="#00838f", lw=1.2))
        ax[0].invert_yaxis(); ax[0].set_aspect("equal")
        ax[0].set_xlabel("x (px)"); ax[0].set_ylabel("y (px)")
        ax[0].set_title("coverage of selected frames", loc="left")
        ax[0].legend(frameon=False, fontsize=7, loc="lower left")
        ax[1].hist(S.frame / 30.0 / 60.0, bins=40, color="#c62828")
        ax[1].set_xlabel("time (min)"); ax[1].set_ylabel("selected frames")
        ax[1].set_title("spread over the session", loc="left")
        fig.tight_layout(); fig.savefig(a.fig, dpi=200)
        print(f"wrote {a.fig}")


if __name__ == "__main__":
    main()
