#!/usr/bin/env python3
"""merge_qc.py — gate, filter and cross-validate DLC keypoints against the blob track.

The two trackers fail in uncorrelated ways: DeepLabCut fails when the animal's
appearance goes out of distribution (cable across the head, miniscope occluding
the ears), the signed-difference blob tracker fails when the animal is motionless
long enough to enter the median reference or when it sits in a dark corner. So
their disagreement is a usable automatic QC signal over hundreds of thousands of
frames that cannot be eyeballed.

What it does
------------
1. Reads the SuperAnimal h5, drops keypoints below `--pcutoff`, median-filters
   each coordinate, and forms a body centre (mouse_center / mid_back / neck,
   first available) and a head point (nose) with head direction.
2. Reads blob_tracker.py output, converts both to cm via the calibration
   homography.
3. Scores per-frame agreement, writes a merged CSV with a `trust` column, and
   renders tracking_qc.png (agreement histogram, per-keypoint confidence,
   occupancy maps from both trackers, speed trace).

Usage
-----
  python merge_qc.py DLC.h5 BLOB.csv -c maze_calib.json -o merged.csv \\
      [--pcutoff 0.6] [--median 5] [--fps 30] [--agree-cm 3.0] [--fig tracking_qc.png]

Deps: numpy, pandas, scipy, matplotlib, tables (for h5), opencv-python
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BODY_PREFS = ("mouse_center", "mid_back", "neck", "mid_backend")
HEAD = "nose"
EARS = ("left_ear", "right_ear")


def read_dlc(h5: Path) -> pd.DataFrame:
    """Return a frame x (bodypart, coord) frame, collapsing scorer/individual levels."""
    df = pd.read_hdf(h5)
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError(f"{h5} is not a DLC multi-index table")
    names = list(df.columns.names)
    for lvl in ("scorer", "individuals"):
        if lvl in names and df.columns.get_level_values(lvl).nunique() == 1:
            df.columns = df.columns.droplevel(lvl)
        elif lvl == "individuals" and lvl in names:
            keep = df.columns.get_level_values(lvl)[0]
            df = df.xs(keep, axis=1, level=lvl)
    df.columns = pd.MultiIndex.from_tuples(
        [tuple(c) for c in df.columns], names=["bodyparts", "coords"])
    return df


def gate_and_filter(dlc: pd.DataFrame, pcut: float, msize: int) -> tuple[pd.DataFrame, pd.Series]:
    parts = dlc.columns.get_level_values("bodyparts").unique()
    out = {}
    conf = {}
    for bp in parts:
        x = dlc[(bp, "x")].to_numpy(float).copy()
        y = dlc[(bp, "y")].to_numpy(float).copy()
        p = dlc[(bp, "likelihood")].to_numpy(float)
        bad = ~np.isfinite(p) | (p < pcut)
        x[bad] = np.nan
        y[bad] = np.nan
        if msize > 1:
            for arr in (x, y):
                ok = np.isfinite(arr)
                if ok.sum() > msize:
                    arr[ok] = median_filter(arr[ok], size=msize, mode="nearest")
        out[(bp, "x")], out[(bp, "y")], out[(bp, "p")] = x, y, p
        conf[bp] = float(np.nanmean(p >= pcut))
    g = pd.DataFrame(out)
    g.columns = pd.MultiIndex.from_tuples(g.columns, names=["bodyparts", "coords"])
    return g, pd.Series(conf, name="frac_above_pcutoff").sort_values()


def body_and_head(g: pd.DataFrame) -> pd.DataFrame:
    parts = list(g.columns.get_level_values("bodyparts").unique())
    body = next((b for b in BODY_PREFS if b in parts), None)
    if body is None:
        raise ValueError(f"none of {BODY_PREFS} present; got {parts}")
    d = pd.DataFrame({"frame": np.arange(len(g)),
                      "dlc_x_px": g[(body, "x")], "dlc_y_px": g[(body, "y")],
                      "dlc_body_part": body,
                      "dlc_body_p": g[(body, "p")]})
    if HEAD in parts and all(e in parts for e in EARS):
        ex = g[[(EARS[0], "x"), (EARS[1], "x")]].mean(axis=1)
        ey = g[[(EARS[0], "y"), (EARS[1], "y")]].mean(axis=1)
        d["head_dir_rad"] = np.arctan2(g[(HEAD, "y")] - ey, g[(HEAD, "x")] - ex)
        d["nose_x_px"], d["nose_y_px"] = g[(HEAD, "x")], g[(HEAD, "y")]
    return d


def to_cm(H: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = np.column_stack([x, y]).astype(np.float32)
    out = np.full_like(xy, np.nan, dtype=float)
    ok = np.isfinite(xy).all(1)
    if ok.any():
        out[ok] = cv2.perspectiveTransform(
            xy[ok].reshape(-1, 1, 2), np.asarray(H, np.float32)).reshape(-1, 2)
    return out[:, 0], out[:, 1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dlc_h5")
    p.add_argument("blob_csv")
    p.add_argument("-c", "--calib", default="maze_calib.json")
    p.add_argument("-o", "--out", default="merged_track.csv")
    p.add_argument("--pcutoff", type=float, default=0.6)
    p.add_argument("--median", type=int, default=5)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--agree-cm", type=float, default=3.0)
    p.add_argument("--fig", default="tracking_qc.png")
    a = p.parse_args()

    calib = json.loads(Path(a.calib).read_text())
    H = np.array(calib["homography_px_to_cm"])
    x0, y0, _, _ = calib["crop_box_px"]

    dlc = read_dlc(Path(a.dlc_h5))
    gated, conf = gate_and_filter(dlc, a.pcutoff, a.median)
    d = body_and_head(gated)
    blob = pd.read_csv(a.blob_csv)

    n = min(len(d), len(blob))
    if len(d) != len(blob):
        print(f"WARNING frame counts differ (dlc {len(d)}, blob {len(blob)}); "
              f"truncating to {n}")
    d, blob = d.iloc[:n].reset_index(drop=True), blob.iloc[:n].reset_index(drop=True)

    # DLC coords are in the cropped frame; lift back to full-frame before the homography
    d["dlc_x_cm"], d["dlc_y_cm"] = to_cm(H, d.dlc_x_px + x0, d.dlc_y_px + y0)
    m = d.copy()
    m["blob_x_cm"], m["blob_y_cm"] = blob.x_cm, blob.y_cm
    m["blob_gated"] = blob.gated
    m["disagree_cm"] = np.hypot(m.dlc_x_cm - m.blob_x_cm, m.dlc_y_cm - m.blob_y_cm)

    has_dlc = np.isfinite(m.dlc_x_cm)
    has_blob = np.isfinite(m.blob_x_cm) & (m.blob_gated == 0)
    agree = m.disagree_cm <= a.agree_cm
    m["trust"] = np.select(
        [has_dlc & has_blob & agree, has_dlc & has_blob & ~agree, has_dlc, has_blob],
        ["both", "conflict", "dlc_only", "blob_only"], default="none")
    # consensus position: mean where both agree, else whichever exists
    m["x_cm"] = np.where(m.trust == "both", (m.dlc_x_cm + m.blob_x_cm) / 2,
                         np.where(has_dlc, m.dlc_x_cm, m.blob_x_cm))
    m["y_cm"] = np.where(m.trust == "both", (m.dlc_y_cm + m.blob_y_cm) / 2,
                         np.where(has_dlc, m.dlc_y_cm, m.blob_y_cm))
    m["speed_cm_s"] = np.hypot(m.x_cm.diff(), m.y_cm.diff()) * a.fps
    m.to_csv(a.out, index=False)

    counts = m.trust.value_counts()
    print(f"\nwrote {a.out}  ({len(m)} frames)")
    print("trust breakdown (fraction of frames):")
    print((counts / len(m)).round(4).to_string())
    print(f"\nmedian DLC-vs-blob disagreement: {m.disagree_cm.median():.2f} cm "
          f"(90th pct {m.disagree_cm.quantile(0.9):.2f} cm)")
    print("\nleast reliable keypoints (fraction of frames above pcutoff):")
    print(conf.head(8).round(3).to_string())

    # ------------------------------------------------------------------ QC figure
    fig, ax = plt.subplots(2, 2, figsize=(9, 7))
    ax[0, 0].hist(m.disagree_cm.dropna(), bins=120, color="#4c72b0")
    ax[0, 0].axvline(a.agree_cm, color="#d62728", ls="--", lw=1)
    ax[0, 0].set_xlabel("DLC - blob distance (cm)")
    ax[0, 0].set_ylabel("frames")
    ax[0, 0].set_title(f"{100 * agree.mean():.1f}% of frames agree within "
                       f"{a.agree_cm:g} cm", loc="left")

    c = conf.head(12)
    ax[0, 1].barh(range(len(c)), c.to_numpy(), color="#55a868")
    ax[0, 1].set_yticks(range(len(c)))
    ax[0, 1].set_yticklabels(c.index, fontsize=7)
    ax[0, 1].set_xlabel(f"fraction of frames with p >= {a.pcutoff:g}")
    ax[0, 1].set_title("Least confident keypoints", loc="left")

    for k, (xc, yc, lab) in enumerate([(m.dlc_x_cm, m.dlc_y_cm, "DeepLabCut"),
                                       (m.blob_x_cm, m.blob_y_cm, "blob tracker")]):
        axx = ax[1, k]
        ok = np.isfinite(xc) & np.isfinite(yc)
        axx.hexbin(xc[ok], yc[ok], gridsize=60, bins="log", cmap="magma")
        axx.set_aspect("equal")
        axx.invert_yaxis()
        axx.set_xlabel("x (cm)")
        axx.set_ylabel("y (cm)")
        axx.set_title(f"{lab} occupancy ({100 * ok.mean():.0f}% of frames)", loc="left")

    fig.tight_layout()
    fig.savefig(a.fig, dpi=200)
    print(f"wrote {a.fig}")


if __name__ == "__main__":
    main()
