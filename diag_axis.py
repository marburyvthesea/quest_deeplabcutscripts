"""Is DLC's displacement anatomical (along the animal) or fixed in the image?

The overlay sheets suggest DLC places body_center anterior to where it was
labelled -- the offset points toward the head and flips with the animal's
heading. That is a keypoint-definition problem. The competing explanation, a
crop-to-frame coordinate defect, displaces in a FIXED IMAGE DIRECTION
regardless of heading. The two make opposite predictions here:

  anatomical  -- the displacement projected on the body axis (tail_base->nose,
    from the human labels) is consistently POSITIVE and large, while the
    perpendicular component scatters around zero. The image-frame direction of
    the offset rotates with the animal.

  coordinate  -- the displacement is consistent in IMAGE coordinates (dx, dy)
    and its projection on the body axis changes sign as the animal turns
    around, averaging toward zero.

Reports both decompositions side by side, plus the distance from DLC's
body_center to each human-labelled part: if the network actually learned a
different part, DLC's body_center should sit closer to that part than to
body_center itself.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import track_benchmark as tb

P = "/projects/b1118/dlc_analysis/TmazeParts-jjm-2026-10-20"
V = "/scratch/jma819/T_maze_recordings/behaviorVIdeos"
PARTS = ["nose", "ear_left", "ear_right", "body_center", "tail_base"]

out = {}
for A in ["C", "F", "H", "K"]:
    try:
        lab = {p: tb.load_labels(f"{P}/labeled-data", A, p) for p in PARTS}
    except Exception as e:                      # H may lack a label file
        print(f"{A}: no labels ({e})")
        continue
    h5 = glob.glob(f"{V}/dlc_out_finetuned/{A}DLC_*.h5")
    if not h5:
        print(f"{A}: no DLC output")
        continue
    df = pd.read_hdf(h5[0])
    sub = df[df.columns.get_level_values(0)[0]]
    if sub.columns.nlevels == 3:
        sub = sub[sub.columns.get_level_values(0)[0]]

    J = lab["body_center"].rename(columns={"xh": "bx", "yh": "by"}).copy()
    for p in PARTS:
        J[f"{p}_x"] = lab[p].xh.reindex(J.index)
        J[f"{p}_y"] = lab[p].yh.reindex(J.index)
    J["dx"] = sub[("body_center", "x")].reindex(J.index).to_numpy(float) - J.bx
    J["dy"] = sub[("body_center", "y")].reindex(J.index).to_numpy(float) - J.by
    J["p"] = sub[("body_center", "likelihood")].reindex(J.index).to_numpy(float)
    J = J[np.isfinite(J.dx)]
    if len(J) < 5:
        print(f"{A}: only {len(J)} usable frames")
        continue

    # body axis from the human labels: tail_base -> nose
    ax_x = J.nose_x - J.tail_base_x
    ax_y = J.nose_y - J.tail_base_y
    L = np.hypot(ax_x, ax_y)
    ux, uy = ax_x / L, ax_y / L
    along = J.dx * ux + J.dy * uy          # + = toward the nose
    perp = -J.dx * uy + J.dy * ux

    rec = {"n": int(len(J)),
           "body_length_px_median": round(float(L.median()), 1),
           "along_axis_median_px": round(float(along.median()), 2),
           "along_axis_frac_positive": round(float((along > 0).mean()), 3),
           "perp_axis_median_px": round(float(perp.median()), 2),
           "image_dx_median_px": round(float(J.dx.median()), 2),
           "image_dy_median_px": round(float(J.dy.median()), 2),
           # spread: the SMALLER spread identifies the frame the offset lives in
           "along_perp_IQR": [round(float(along.quantile(.75) - along.quantile(.25)), 1),
                              round(float(perp.quantile(.75) - perp.quantile(.25)), 1)],
           "image_dxdy_IQR": [round(float(J.dx.quantile(.75) - J.dx.quantile(.25)), 1),
                              round(float(J.dy.quantile(.75) - J.dy.quantile(.25)), 1)]}

    # which labelled part is DLC's body_center actually closest to?
    dlcx, dlcy = J.bx + J.dx, J.by + J.dy
    rec["median_dist_to_each_labelled_part"] = {
        p: round(float(np.hypot(dlcx - J[f"{p}_x"], dlcy - J[f"{p}_y"]).median()), 2)
        for p in PARTS}
    out[A] = rec
    print(f"===== {A}")
    print(json.dumps(rec, indent=1))

json.dump(out, open("axis_diag.json", "w"), indent=1)
