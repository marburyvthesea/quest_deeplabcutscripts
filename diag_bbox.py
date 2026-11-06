"""Do partially-labelled frames produce degenerate detector boxes?

The census found 107 of 399 labelled frames carry at least one NaN keypoint,
and that DLC kept exactly the 276 frames with none. For a top-down model the
detector's training boxes are derived from the labelled keypoints, so a frame
with only the nose labelled yields a box a few pixels across. A detector
trained on a mix of body-sized and keypoint-sized boxes learns an incoherent
notion of scale -- which is what sparse, low-confidence, badly-framed
detections look like at inference.

This measures the boxes the keypoints imply, split by how complete the label
is, and asks whether the incomplete ones are degenerate. It reads only the
merged CollectedData frame -- no model, no video, seconds to run.

  python diag_bbox.py                    # default project and margin
  python diag_bbox.py --margin 0         # raw keypoint extent
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

P = "/projects/b1118/dlc_analysis/TmazeParts-jjm-2026-10-20"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=P)
    ap.add_argument("--margin", type=float, default=20.0,
                    help="px added on each side, as DLC does when building "
                         "detector boxes from keypoints")
    ap.add_argument("--fig", default="bbox_audit.png")
    a = ap.parse_args()

    hits = sorted(glob.glob(
        f"{a.project}/training-datasets/*/*/CollectedData_*.h5"))
    if not hits:
        sys.exit(f"no merged CollectedData_*.h5 under {a.project}")
    D = pd.read_hdf(hits[0])
    scorer = D.columns.get_level_values(0)[0]
    parts = list(dict.fromkeys(D.columns.get_level_values(-2)))
    print(f"{os.path.basename(hits[0])}: {len(D)} frames, parts {parts}")

    X = np.stack([D[(scorer, p, "x")].to_numpy(float) for p in parts], 1)
    Y = np.stack([D[(scorer, p, "y")].to_numpy(float) for p in parts], 1)
    vis = np.isfinite(X) & np.isfinite(Y)
    nvis = vis.sum(1)

    with np.errstate(invalid="ignore"):
        w = np.nanmax(np.where(vis, X, np.nan), 1) - \
            np.nanmin(np.where(vis, X, np.nan), 1) + 2 * a.margin
        h = np.nanmax(np.where(vis, Y, np.nan), 1) - \
            np.nanmin(np.where(vis, Y, np.nan), 1) + 2 * a.margin
    diag = np.hypot(w, h)
    animal = np.array([t[-2] if isinstance(t, tuple) else "?" for t in D.index])

    print(f"\n=== label completeness (margin {a.margin:g} px)")
    rows = []
    for k in range(len(parts) + 1):
        m = nvis == k
        if not m.any():
            continue
        rows.append(dict(n_parts=k, frames=int(m.sum()),
                         med_w=round(float(np.nanmedian(w[m])), 1),
                         med_h=round(float(np.nanmedian(h[m])), 1),
                         med_diag=round(float(np.nanmedian(diag[m])), 1),
                         min_diag=round(float(np.nanmin(diag[m])), 1)))
    T = pd.DataFrame(rows)
    print(T.to_string(index=False))

    full, part = nvis == len(parts), (nvis > 0) & (nvis < len(parts))
    print(f"\ncomplete labels: {int(full.sum())} frames, "
          f"median box {np.nanmedian(diag[full]):.1f} px diagonal")
    if part.any():
        print(f"partial labels:  {int(part.sum())} frames, "
              f"median box {np.nanmedian(diag[part]):.1f} px diagonal "
              f"({100*np.nanmedian(diag[part])/np.nanmedian(diag[full]):.0f}% "
              "of complete)")
        tiny = part & (diag < 0.5 * np.nanmedian(diag[full]))
        print(f"boxes under half the typical size: {int(tiny.sum())} "
              f"({100*tiny.mean():.1f}% of all frames)")
    print("\nper-animal partial-label counts:",
          {k: int(v) for k, v in
           pd.Series(animal[part]).value_counts().items()} if part.any() else {})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    bins = np.linspace(0, np.nanpercentile(diag, 99.5), 40)
    ax[0].hist(diag[full], bins=bins, color="#4878a8", label="all parts")
    if part.any():
        ax[0].hist(diag[part], bins=bins, color="#c0504d", alpha=.75,
                   label="incomplete")
    ax[0].set_xlabel("implied box diagonal (px)")
    ax[0].set_ylabel("frames")
    ax[0].legend(frameon=False)
    ax[1].scatter(nvis + np.random.default_rng(0).normal(0, .06, len(nvis)),
                  diag, s=9, alpha=.5, color="#4878a8")
    ax[1].set_xlabel("labelled parts in frame")
    ax[1].set_ylabel("implied box diagonal (px)")
    for s in ("top", "right"):
        for b in ax:
            b.spines[s].set_visible(False)
    fig.suptitle("Detector boxes implied by the labels", x=.02, ha="left")
    fig.tight_layout()
    fig.savefig(a.fig, dpi=160)
    print(f"\nwrote {a.fig}")
    T.to_csv("bbox_audit.csv", index=False)


if __name__ == "__main__":
    main()
