"""Draw human label, DLC prediction and blob centroid on the same frames.

This resolves the one thing the numeric benchmark could not: DLC sits ~13 px
left of both the human labels and the blob track, and whether that is a
differently-placed anatomical point or a coordinate-transform defect is
visible at a glance but invisible in a summary statistic.

Renders a contact sheet of labelled frames -- the frames where all three
sources have a position -- cropped around the animal so the displacement is
legible. Green = human label, blue = DLC, red = blob.

  python overlay_check.py C --n 12 --out C_overlay.png

Full-session labelled videos are deliberately not what this does: a
250k-frame render answers the same question far more slowly.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import track_benchmark as tb

P = "/projects/b1118/dlc_analysis/TmazeParts-jjm-2026-10-20"
V = "/scratch/jma819/T_maze_recordings/behaviorVIdeos"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("animal")
    ap.add_argument("--bodypart", default="body_center")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--pad", type=int, default=90, help="crop half-width (px)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"{a.animal}_overlay.png"

    lab = tb.load_labels(f"{P}/labeled-data", a.animal, a.bodypart)
    h5 = glob.glob(f"{V}/dlc_out_finetuned/{a.animal}DLC_*.h5")[0]
    df = pd.read_hdf(h5)
    sub = df[df.columns.get_level_values(0)[0]]
    if sub.columns.nlevels == 3:
        sub = sub[sub.columns.get_level_values(0)[0]]
    pred = pd.DataFrame({"xd": sub[(a.bodypart, "x")].to_numpy(float),
                         "yd": sub[(a.bodypart, "y")].to_numpy(float),
                         "p": sub[(a.bodypart, "likelihood")].to_numpy(float)},
                        index=np.arange(len(sub)))
    blob = tb.load_blob(f"{V}/{a.animal}_blobtrack.csv", True)

    J = lab.join(pred, how="left").join(blob, how="left")
    J = J[np.isfinite(J.xd) & np.isfinite(J.xh)]
    if not len(J):
        sys.exit(f"no labelled frame of {a.animal} has a DLC position")
    # prefer the most confident frames: the displacement is largest there
    J = J.sort_values("p", ascending=False).head(a.n).sort_index()

    cap = cv2.VideoCapture(f"{V}/{a.animal}.avi")
    nc = int(np.ceil(np.sqrt(len(J))))
    nr = int(np.ceil(len(J) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.6 * nr))
    axs = np.atleast_1d(axs).ravel()

    for ax, (fr, r) in zip(axs, J.iterrows()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
        ok, im = cap.read()
        if not ok:
            ax.remove()
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        cx, cy = r.xh, r.yh
        x0, x1 = int(max(0, cx - a.pad)), int(min(im.shape[1], cx + a.pad))
        y0, y1 = int(max(0, cy - a.pad)), int(min(im.shape[0], cy + a.pad))
        ax.imshow(im[y0:y1, x0:x1])
        ax.plot(r.xh - x0, r.yh - y0, "o", mfc="none", mec="#2ca02c", ms=11, mew=2)
        ax.plot(r.xd - x0, r.yd - y0, "+", color="#1f6feb", ms=13, mew=2)
        if np.isfinite(r.get("xb", np.nan)):
            ax.plot(r.xb - x0, r.yb - y0, "x", color="#d1495b", ms=11, mew=2)
        ax.set_title(f"frame {fr}  p={r.p:.2f}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axs[len(J):]:
        ax.remove()

    fig.suptitle(f"{a.animal}: human (green o), DLC (blue +), blob (red x)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    cap.release()
    print(f"wrote {out} with {len(J)} frames")


if __name__ == "__main__":
    main()
