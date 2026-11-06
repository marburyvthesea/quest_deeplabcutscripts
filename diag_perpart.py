"""Which SuperAnimal channels did analyze_videos actually write?

evaluate_network crashes on this project with

  pred["bodyparts"] = pred["bodyparts"][:, weight_init.conversion_array]
  IndexError: index 9 is out of bounds for axis 1 with size 5

so DLC applies conversion_array=[0,1,2,9,13] to an array that already has
only 5 columns -- the 27 head channels have been reduced before that line
runs. analyze_videos goes through related code and did not crash, so either
it received 27 channels and selected correctly, or it received 5 and those
5 are channels 0-4 of the SuperAnimal skeleton rather than 0,1,2,9,13.

Channels 0,1,2 are nose and the two ears under either reading. They diverge
at the last two: the project expects body_center and tail_base (channels 9
and 13), whereas channels 3 and 4 are ear tips. So a positional mis-mapping
has a signature that needs no GPU and no model:

  nose/ear_left/ear_right accurate, body_center/tail_base badly displaced
      -> the conversion was skipped; the last two columns are ear tips
  all five comparable
      -> the conversion was applied correctly and the error is genuine

The cross-error matrix is the direct form of the same question: the median
distance from every predicted column to every labelled part. If predicted
body_center sits closer to a labelled ear than to the labelled body_center,
the mapping is shifted and the diagonal will not be the minimum.

Usage:
  python diag_perpart.py
  python diag_perpart.py --project ... --dir ...
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

P = "/projects/b1118/dlc_analysis/TmazeParts-jjm-2026-10-20"
D = "/scratch/jma819/T_maze_recordings/behaviorVIdeos/dlc_out_finetuned"
ORDER = ["nose", "ear_left", "ear_right", "body_center", "tail_base"]


def top(df):
    return df[df.columns.get_level_values(0)[0]]


def frame_of(ix):
    fn = ix[-1] if isinstance(ix, tuple) else os.path.basename(str(ix))
    digits = "".join(ch for ch in fn if ch.isdigit())
    return int(digits) if digits else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=P)
    ap.add_argument("--dir", default=D)
    ap.add_argument("--out", default="perpart_err.csv")
    a = ap.parse_args()

    rows, cross = [], []
    for d in sorted(glob.glob(os.path.join(a.project, "labeled-data", "*"))):
        v = os.path.basename(d)
        lp = glob.glob(os.path.join(d, "CollectedData_*.h5"))
        vp = sorted(glob.glob(os.path.join(a.dir, v + "DLC*.h5")))
        if not lp or not vp:
            continue
        print(f"  reading {v} ...", flush=True)
        L, V = top(pd.read_hdf(lp[0])), top(pd.read_hdf(vp[0]))
        lparts = [p for p in ORDER if p in L.columns.get_level_values(0)]
        vparts = [p for p in ORDER if p in V.columns.get_level_values(0)]
        n_v = len(V)

        for ix in L.index:
            fr = frame_of(ix)
            if fr < 0 or fr >= n_v:
                continue
            row = V.iloc[fr]
            lab = {p: (float(L.loc[ix, (p, "x")]), float(L.loc[ix, (p, "y")]))
                   for p in lparts}
            pred = {p: (float(row[(p, "x")]), float(row[(p, "y")]),
                        float(row[(p, "likelihood")])) for p in vparts}
            for p in vparts:
                px, py, pl = pred[p]
                if p in lab and np.isfinite([px, py, *lab[p]]).all():
                    rows.append(dict(video=v, bodypart=p, p=pl,
                                     err=float(np.hypot(px - lab[p][0],
                                                        py - lab[p][1]))))
                for q in lparts:
                    lx, ly = lab[q]
                    if np.isfinite([px, py, lx, ly]).all():
                        cross.append(dict(pred=p, label=q,
                                          d=float(np.hypot(px - lx, py - ly))))

    R, X = pd.DataFrame(rows), pd.DataFrame(cross)
    if R.empty:
        raise SystemExit("no labelled frames matched the analysis output")
    R.to_csv(a.out, index=False)

    print(f"\nn = {len(R)} predicted-vs-labelled comparisons\n")
    g = R.groupby("bodypart").agg(
        n=("err", "size"), median_px=("err", "median"),
        p25=("err", lambda s: s.quantile(.25)),
        p75=("err", lambda s: s.quantile(.75)),
        median_p=("p", "median")).round(2)
    g = g.reindex([o for o in ORDER if o in g.index])
    print("--- error of each predicted column against its own label")
    print(g.to_string())

    print("\n--- median px per video x bodypart")
    print(R.pivot_table(index="video", columns="bodypart", values="err",
                        aggfunc="median").round(1)
           .reindex(columns=[o for o in ORDER if o in set(R.bodypart)]).to_string())

    print("\n--- cross-error matrix: median px from predicted (row) to labelled (col)")
    M = X.pivot_table(index="pred", columns="label", values="d", aggfunc="median")
    M = M.reindex(index=[o for o in ORDER if o in M.index],
                  columns=[o for o in ORDER if o in M.columns])
    print(M.round(1).to_string())

    print("\n--- nearest labelled part for each predicted column")
    shifted = []
    for p in M.index:
        best = M.loc[p].idxmin()
        flag = "" if best == p else f"   <-- NOT its own label"
        print(f"  predicted {p:12s} -> closest to labelled {best:12s} "
              f"({M.loc[p, best]:.1f} px vs {M.loc[p, p]:.1f} px own){flag}")
        if best != p:
            shifted.append(p)

    head = [b for b in ("nose", "ear_left", "ear_right") if b in g.index]
    tail = [b for b in ("body_center", "tail_base") if b in g.index]
    print("")
    if head and tail:
        mh = g.loc[head, "median_px"].median()
        mt = g.loc[tail, "median_px"].median()
        print(f"nose+ears median {mh:.1f} px | body_center+tail_base median {mt:.1f} px"
              f" | ratio {mt/max(mh,1e-9):.2f}")
        if mt > 2.5 * mh:
            print("VERDICT: the first three columns are accurate and the last two are "
                  "not. That is the signature of channels 0-4 being written in place "
                  "of 0,1,2,9,13 -- body_center and tail_base are carrying ear tips. "
                  "The network is fine; analyze_videos wrote the wrong channels.")
        elif shifted:
            print("VERDICT: errors are comparable across parts, but %s do not sit "
                  "closest to their own label -- inspect the cross-error matrix."
                  % ", ".join(shifted))
        else:
            print("VERDICT: no positional mis-mapping. Every column is closest to its "
                  "own label and the head/tail errors are comparable, so the 18 px is "
                  "genuine pose error.")

    print("\n=== SuperAnimal topviewmouse keypoint order, for reference")
    got = False
    for mod, fn in (("deeplabcut.modelzoo.utils", "get_super_animal_project_cfg"),
                    ("deeplabcut.modelzoo.utils", "read_config_as_dict"),
                    ("deeplabcut.modelzoo.generalized_data_converter.datasets",
                     "get_super_animal_project_cfg")):
        try:
            m = __import__(mod, fromlist=[fn])
            kp = getattr(m, fn)("superanimal_topviewmouse")
            kp = kp["bodyparts"] if isinstance(kp, dict) else kp
            for i, k in enumerate(list(kp)[:15]):
                tag = "   <-- in conversion_array" if i in (0, 1, 2, 9, 13) else ""
                print(f"  {i:2d} {k}{tag}")
            got = True
            break
        except Exception:
            continue
    if not got:
        print("  (could not load the modelzoo config; channels 0,1,2 are nose and the "
              "two ears, 9 and 13 are the body centre and tail base)")

    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
