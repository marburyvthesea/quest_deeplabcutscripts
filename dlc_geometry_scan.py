#!/usr/bin/env python
"""
Classify every frame of a DeepLabCut prediction file against maze geometry.

The maze is a rectangle with two internal blocks the animal cannot occupy, so a
predicted body-part inside a block, or outside the rectangle, is wrong by
construction. This gives a label-free accuracy signal and, more usefully, a
list of frames the network already gets right -- candidates for a training set.

  python dlc_geometry_scan.py --h5 C_...h5 --geom maze_geometry_C.json \
         --pcutoff 0.6 --out C_framescan.csv --fig C_framescan.png

Status per frame, for the best-scoring individual:
  ok       primary keypoint confident, inside the maze, outside both blocks
  island   confident but inside an excluded block
  outside  confident but outside the maze rectangle
  lowp     primary keypoint below the likelihood cutoff
"""
import argparse
import json
import numpy as np
import pandas as pd


def load_h5(path):
    d = pd.read_hdf(path)
    names = list(d.columns.names or [])
    if "scorer" in names:
        d.columns = d.columns.droplevel("scorer")
        names.remove("scorer")
    if "individuals" not in names:
        d = pd.concat({"animal0": d}, axis=1, names=["individuals"])
    return d.reorder_levels(["individuals", "bodyparts", "coords"], axis=1).sort_index(axis=1)


def in_box(x, y, b, pad=0.0):
    return ((x >= b["x"] - pad) & (x <= b["x"] + b["w"] + pad)
            & (y >= b["y"] - pad) & (y <= b["y"] + b["h"] + pad))


def runs(mask):
    """[(start, stop_exclusive, length), ...] for each run of True."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    e = np.flatnonzero(np.diff(np.concatenate(([0], m.view(np.int8), [0]))))
    return [(int(a), int(b), int(b - a)) for a, b in zip(e[::2], e[1::2])]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", required=True)
    p.add_argument("--geom", required=True)
    p.add_argument("--pcutoff", type=float, default=0.6)
    p.add_argument("--primary", default="mouse_center")
    p.add_argument("--pad", type=float, default=0.0,
                   help="px tolerance on the block/boundary tests")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--out", required=True)
    p.add_argument("--fig", default=None)
    a = p.parse_args()

    g = json.load(open(a.geom))
    outer, islands = g["maze_outer"], g["exclude_islands"]
    d = load_h5(a.h5)
    n = len(d)
    inds = list(d.columns.get_level_values("individuals").unique())
    bps = list(d[inds[0]].columns.get_level_values("bodyparts").unique())
    key = a.primary if a.primary in bps else bps[0]

    # pick, per frame, the individual whose primary keypoint scores highest
    P = np.column_stack([d[(i, key, "likelihood")].to_numpy(float) for i in inds])
    P = np.nan_to_num(P, nan=-1.0)
    best = P.argmax(1)
    take = lambda c: np.column_stack(
        [d[(i, key, c)].to_numpy(float) for i in inds])[np.arange(n), best]
    x, y, pk = take("x"), take("y"), P[np.arange(n), best]

    inmaze = in_box(x, y, outer, a.pad)
    inisle = np.zeros(n, bool)
    for b in islands:
        inisle |= in_box(x, y, b, -a.pad)

    status = np.full(n, "lowp", dtype=object)
    conf = pk >= a.pcutoff
    status[conf & inmaze & ~inisle] = "ok"
    status[conf & inisle] = "island"
    status[conf & ~inmaze] = "outside"

    # how much of the whole skeleton is geometrically plausible, per frame
    kp_ok = np.zeros(n), np.zeros(n)
    tot = np.zeros(n)
    good = np.zeros(n)
    for b in bps:
        bx = np.column_stack([d[(i, b, "x")].to_numpy(float) for i in inds])[np.arange(n), best]
        by = np.column_stack([d[(i, b, "y")].to_numpy(float) for i in inds])[np.arange(n), best]
        bp = np.column_stack([d[(i, b, "likelihood")].to_numpy(float) for i in inds])
        bp = np.nan_to_num(bp, nan=0.0)[np.arange(n), best]
        c = bp >= a.pcutoff
        tot += c
        good += c & in_box(bx, by, outer, a.pad) & ~np.any(
            [in_box(bx, by, bb, -a.pad) for bb in islands], axis=0)

    out = pd.DataFrame(dict(frame=np.arange(n), individual=[inds[i] for i in best],
                            x=x, y=y, p=pk, status=status,
                            n_conf_kp=tot.astype(int),
                            frac_kp_valid=np.where(tot > 0, good / np.maximum(tot, 1), np.nan)))
    out.to_csv(a.out, index=False)

    vc = out.status.value_counts()
    print(f"frames: {n}   individuals: {inds}   primary keypoint: {key}")
    for k in ("ok", "island", "outside", "lowp"):
        print(f"  {k:8s} {int(vc.get(k, 0)):7d}  {100*vc.get(k,0)/n:5.1f}%")
    ok = (out.status == "ok").to_numpy()
    rr = runs(ok)
    lens = np.array([r[2] for r in rr]) if rr else np.array([0])
    print(f"\ncontiguous 'ok' runs: {len(rr)}; longest {lens.max()} frames "
          f"({lens.max()/a.fps:.1f} s); median {np.median(lens):.0f}; "
          f"runs >= 30 frames: {int((lens >= 30).sum())}")
    print("top 10 runs (start, stop, length):")
    for r in sorted(rr, key=lambda r: -r[2])[:10]:
        print(f"  {r[0]:7d} {r[1]:7d} {r[2]:6d}")
    strict = ok & (out.frac_kp_valid >= 0.9).to_numpy()
    print(f"\nframes with primary ok AND >=90% of confident keypoints plausible: "
          f"{int(strict.sum())} ({100*strict.mean():.1f}%)")
    print(f"wrote {a.out}")

    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
        cols = {"ok": "#4c72b0", "island": "#dd8452", "outside": "#c44e52", "lowp": "#9a9a9a"}
        for s, c in cols.items():
            m = (out.status == s).to_numpy()
            ax[0].plot(out.frame[m], np.full(m.sum(), s), "|", color=c, ms=4, alpha=.5)
        ax[0].set_xlabel("frame"); ax[0].set_title("status over the session", loc="left")
        for s in ("island", "outside", "ok"):
            m = (out.status == s).to_numpy()
            ax[1].plot(out.x[m], out.y[m], ".", ms=1, color=cols[s], alpha=.25, label=s)
        for b in islands + [outer]:
            ax[1].add_patch(plt.Rectangle((b["x"], b["y"]), b["w"], b["h"],
                                          fill=False, ec="#333333", lw=1))
        ax[1].invert_yaxis(); ax[1].set_aspect("equal")
        ax[1].set_xlabel("x (px)"); ax[1].set_ylabel("y (px)")
        ax[1].legend(frameon=False, markerscale=8, fontsize=7)
        ax[1].set_title(f"{key} predictions", loc="left")
        if rr:
            ax[2].hist(lens, bins=np.logspace(0, np.log10(max(lens.max(), 2)), 40),
                       color="#4c72b0")
            ax[2].set_xscale("log")
        ax[2].set_xlabel("length of contiguous 'ok' run (frames)")
        ax[2].set_ylabel("count"); ax[2].set_title("usable stretches", loc="left")
        fig.tight_layout(); fig.savefig(a.fig, dpi=200)
        print(f"wrote {a.fig}")


if __name__ == "__main__":
    main()
