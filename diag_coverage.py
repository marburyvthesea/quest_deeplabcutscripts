"""Detection coverage per video -- is the failure global or regime-specific?

Every coverage figure so far (~22 %) comes from C, F and K. The contact sheet
showed those three are a different imaging regime from the five *_test
recordings: 2.4x brighter, with 29 % of pixels clipped to pure white against
0.8 % for the *_test videos, and no overlap at all between the two groups.

So the 22 % may describe the detector, or it may describe only the saturated
regime. The distinction decides what the 107 relabelled frames can buy:

  *_test coverage also poor -> the detector is broken across the board and
        the unlabelled frames entering the split as negatives is the leading
        explanation. Relabel, rebuild the training set, retrain.
  *_test coverage good      -> the detector works where the image is not
        blown out, and the failure is specific to C/F/K. 47 more labels there
        will help but will not restore clipped pixels; exposure is the root
        cause and no amount of labelling fixes a saturated sensor.

Reads only the analysis .h5 files already on disk -- no model, no video.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

V = "/scratch/jma819/T_maze_recordings/behaviorVIdeos"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=f"{V}/dlc_out_finetuned")
    ap.add_argument("--bodypart", default="body_center")
    ap.add_argument("--pcutoff", type=float, default=0.6)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.h5")))
    if not files:
        raise SystemExit(f"no .h5 under {a.dir}")

    rows = []
    for f in files:
        base = os.path.basename(f)
        video = re.split(r"DLC_", base)[0]
        try:
            df = pd.read_hdf(f)
        except Exception as e:
            print(f"  {base}: unreadable ({e})")
            continue
        sub = df[df.columns.get_level_values(0)[0]]
        if sub.columns.nlevels == 3:
            sub = sub[sub.columns.get_level_values(0)[0]]
        parts = list(dict.fromkeys(sub.columns.get_level_values(0)))
        bp = a.bodypart if a.bodypart in parts else parts[0]
        x = sub[(bp, "x")].to_numpy(float)
        y = sub[(bp, "y")].to_numpy(float)
        p = sub[(bp, "likelihood")].to_numpy(float)
        # both encodings of "no detection" have to count
        miss = ~np.isfinite(x) | ~np.isfinite(y) | ((x == -1) & (y == -1))
        det = ~miss
        rows.append(dict(video=video, part=bp, frames=len(x),
                         coverage_pct=round(100 * det.mean(), 1),
                         above_p_pct=round(100 * (det & (p >= a.pcutoff)).mean(), 1),
                         median_p=round(float(np.nanmedian(p[det])), 3)
                         if det.any() else np.nan))

    T = pd.DataFrame(rows).sort_values("coverage_pct")
    print(T.to_string(index=False))
    T.to_csv("coverage_by_video.csv", index=False)

    # the C/F/K recordings are the saturated regime; *_test are not
    sat = T.video.str.fullmatch(r"[CFHK]")
    if sat.any() and (~sat).any():
        print(f"\nsaturated regime (C/F/H/K): "
              f"{T[sat].coverage_pct.mean():.1f}% mean coverage")
        print(f"unsaturated (*_test):        "
              f"{T[~sat].coverage_pct.mean():.1f}% mean coverage")
    else:
        missing = "*_test" if not (~sat).any() else "C/F/H/K"
        print(f"\nonly one regime present in {a.dir} -- no {missing} outputs. "
              "Run analyze_videos on one *_test recording to compare.")
    print("\nwrote coverage_by_video.csv")


if __name__ == "__main__":
    main()
