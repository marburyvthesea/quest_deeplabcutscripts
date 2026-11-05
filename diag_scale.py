"""Is DLC's offset constant, or does it grow with position?

This distinguishes two diagnoses that the previous checks cannot separate:

  scale mismatch -- analyze_videos ran at a different resolution than the
    labelled images, so predictions are true*s. Then dx grows linearly with x
    and dy with y, and fitting s recovers a single factor that explains both.
    This is a configuration bug and the network itself may be fine.

  the network is off -- dx is flat in x, i.e. a genuine constant displacement
    or genuine error, and no rescaling rescues it.

Fits dx = (s-1)*x + b on frames where both DLC and blob report a position,
and reports what the residual error would be after applying the fitted scale.
If the residual drops to the blob tracker's own ~8 px accuracy, the mismatch
is the whole story.
"""
import glob
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jma819/quest_deeplabcutscripts")
import track_benchmark as tb

V = "/scratch/jma819/T_maze_recordings/behaviorVIdeos"
BP = "body_center"

out = {}
for A in ["C", "F", "K"]:
    h5 = glob.glob(f"{V}/dlc_out_finetuned/{A}DLC_*.h5")[0]
    df = pd.read_hdf(h5)
    sub = df[df.columns.get_level_values(0)[0]]
    if sub.columns.nlevels == 3:
        sub = sub[sub.columns.get_level_values(0)[0]]
    pred = pd.DataFrame({"xd": sub[(BP, "x")].to_numpy(float),
                         "yd": sub[(BP, "y")].to_numpy(float),
                         "p": sub[(BP, "likelihood")].to_numpy(float)},
                        index=np.arange(len(sub)))
    blob = tb.load_blob(f"{V}/{A}_blobtrack.csv", True)
    J = pred.join(blob, how="inner").dropna(subset=["xd", "yd", "xb", "yb"])
    J = J[J.p > 0.6]                       # confident frames: where the offset is largest

    rec = {"n": int(len(J))}
    for ax, dcol, bcol in [("x", "xd", "xb"), ("y", "yd", "yb")]:
        d = J[dcol] - J[bcol]
        sl, ic = np.polyfit(J[bcol], d, 1)
        rec[f"{ax}_slope"] = round(float(sl), 4)       # s-1
        rec[f"{ax}_intercept"] = round(float(ic), 2)
        rec[f"{ax}_implied_scale"] = round(float(1 + sl), 4)
        # offset in the lowest and highest decile of position
        lo, hi = J[bcol].quantile(.1), J[bcol].quantile(.9)
        rec[f"{ax}_offset_at_low"] = round(float(d[J[bcol] <= lo].median()), 2)
        rec[f"{ax}_offset_at_high"] = round(float(d[J[bcol] >= hi].median()), 2)

    # what would the error be after undoing the fitted affine map?
    xc = (J.xd - rec["x_intercept"]) / rec["x_implied_scale"]
    yc = (J.yd - rec["y_intercept"]) / rec["y_implied_scale"]
    rec["median_px_raw"] = round(float(np.hypot(J.xd - J.xb, J.yd - J.yb).median()), 2)
    rec["median_px_after_rescale"] = round(float(np.hypot(xc - J.xb, yc - J.yb).median()), 2)
    out[A] = rec
    print(f"===== {A}")
    print(json.dumps(rec, indent=1))

json.dump(out, open("scale_diag.json", "w"), indent=1)
