"""Rebuild the analysis h5 from the _full.pickle using the correct channels.

THE BUG. The Oct-20 network is a SuperAnimal memory-replay fine-tune: the pose
head emits all 27 superanimal_topviewmouse keypoints and the five project parts
are channels [0, 1, 2, 9, 13] via conversion_array. analyze_videos wrote the
h5 by taking the FIRST five channels instead of converting, so the columns hold
superanimal channels 0-4. Channels 0,1,2 are nose and the two ears, which is
why those columns are accurate; channels 3 and 4 are the ear TIPS, which went
into body_center and tail_base.

Measured on 1742 labelled frames (diag_perpart.py):

  nose          3.43 px      ear_left   4.72 px      ear_right  4.65 px
  body_center  17.06 px      tail_base 29.62 px

and the cross-error matrix puts predicted body_center 9.0 px from the labelled
LEFT EAR (vs 17.1 px from the labelled body centre) and predicted tail_base
11.6 px from the labelled RIGHT EAR (vs 29.6 px from the tail base).

The same double-conversion bug makes evaluate_network crash outright:
  pred["bodyparts"] = pred["bodyparts"][:, weight_init.conversion_array]
  IndexError: index 9 is out of bounds for axis 1 with size 5
because model.py:219 already applied the conversion inside the model.

WHY NO RETRAIN IS NEEDED. The _full.pickle files kept every channel --
coordinates (1, 27, 1, 2) and confidence (27, 1, 1) per frame -- so the correct
body_center and tail_base are already on disk for every frame of every video.
This reads them out and writes a corrected h5. No GPU, no inference, no
retraining: the network was always right, only the written output was wrong.

Originals are never overwritten; corrected files get a suffix. Where human
labels exist the script validates the result against them and prints the
before/after error, so a silent mis-selection cannot pass unnoticed.

  python rebuild_h5_from_pickle.py --videos 311_test      # one video first
  python rebuild_h5_from_pickle.py                        # all of them
"""

import argparse
import glob
import os
import pickle
import re

import numpy as np
import pandas as pd

P = "/projects/b1118/dlc_analysis/TmazeParts-jjm-2026-10-20"
D = "/scratch/jma819/T_maze_recordings/behaviorVIdeos/dlc_out_finetuned"
FALLBACK_CONV = [0, 1, 2, 9, 13]


def find_conversion_array(project):
    """Read conversion_array from the training config rather than assuming it."""
    pats = os.path.join(project, "dlc-models-pytorch", "iteration-*", "*",
                        "train", "pytorch_config.yaml")
    for p in sorted(glob.glob(pats)):
        try:
            import yaml
            cfg = yaml.safe_load(open(p))
            wi = cfg.get("train_settings", {}).get("weight_init") or {}
            ca = wi.get("conversion_array")
            if ca is not None:
                return [int(v) for v in ca], p
        except Exception:
            continue
    return None, None


def project_bodyparts(project):
    try:
        import yaml
        cfg = yaml.safe_load(open(os.path.join(project, "config.yaml")))
        bp = cfg.get("multianimalbodyparts") or cfg.get("bodyparts")
        return list(bp) if bp else None
    except Exception:
        return None


def frame_index(key):
    m = re.search(r"(\d+)$", key)
    return int(m.group(1)) if m else -1


def read_pickle_channels(path):
    """-> xy (n, K, 2), p (n, K). Frames without a detection become NaN."""
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    keys = [k for k in d if k != "metadata"]
    keys.sort(key=frame_index)
    n = len(keys)
    K = None
    for k in keys[:50]:
        c = np.asarray(d[k].get("coordinates"))
        if c.size:
            K = np.squeeze(c).reshape(-1, 2).shape[0]
            break
    if K is None:
        raise SystemExit(f"{os.path.basename(path)}: no frame carries coordinates")

    xy = np.full((n, K, 2), np.nan)
    pp = np.full((n, K), np.nan)
    for i, k in enumerate(keys):
        v = d[k]
        c = np.asarray(v.get("coordinates"))
        if c.size:
            s = np.squeeze(c).reshape(-1, 2)
            if s.shape[0] == K:
                xy[i] = s
        q = np.asarray(v.get("confidence"))
        if q.size:
            s = np.squeeze(q).reshape(-1)
            if s.shape[0] == K:
                pp[i] = s
    return xy, pp, K


def original_scorer_and_key(h5_path):
    scorer, key = None, "df_with_missing"
    if h5_path and os.path.exists(h5_path):
        try:
            with pd.HDFStore(h5_path, "r") as st:
                ks = [k.strip("/") for k in st.keys()]
                if ks:
                    key = ks[0]
            scorer = pd.read_hdf(h5_path).columns.get_level_values(0)[0]
        except Exception:
            pass
    return scorer, key


def top(df):
    return df[df.columns.get_level_values(0)[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=P)
    ap.add_argument("--dir", default=D)
    ap.add_argument("--videos", nargs="*", default=None,
                    help="restrict to these video names")
    ap.add_argument("--suffix", default="_corrected")
    ap.add_argument("--csv", action="store_true", help="also write a csv")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo corrected files that already exist")
    a = ap.parse_args()

    conv, conv_src = find_conversion_array(a.project)
    if conv is None:
        conv, conv_src = FALLBACK_CONV, "(fallback -- config not found)"
    bps = project_bodyparts(a.project) or ["bp%d" % i for i in range(len(conv))]
    print(f"conversion_array = {conv}")
    print(f"  from {conv_src}")
    print(f"project bodyparts = {bps}")
    if len(bps) != len(conv):
        raise SystemExit(f"bodyparts ({len(bps)}) and conversion_array "
                         f"({len(conv)}) disagree -- refusing to guess")

    pkls = sorted(glob.glob(os.path.join(a.dir, "*_full.pickle")))
    if a.videos:
        pkls = [p for p in pkls
                if any(os.path.basename(p).startswith(v + "DLC") for v in a.videos)]
    if not pkls:
        raise SystemExit(f"no *_full.pickle under {a.dir}")

    summary = []
    for pk in pkls:
        base = os.path.basename(pk)[:-len("_full.pickle")]
        video = re.split(r"DLC_", base)[0].rstrip("_")
        out_h5 = os.path.join(a.dir, base + a.suffix + ".h5")
        if os.path.exists(out_h5) and not a.overwrite:
            print(f"\n{video}: {os.path.basename(out_h5)} exists -- skip "
                  "(pass --overwrite to redo)")
            continue

        print(f"\n{video}: reading {base}_full.pickle ...", flush=True)
        xy, pp, K = read_pickle_channels(pk)
        print(f"  {xy.shape[0]} frames x {K} channels")
        if max(conv) >= K:
            raise SystemExit(f"  conversion_array needs channel {max(conv)} but the "
                             f"pickle has only {K} -- cannot repair this file")

        orig_h5 = os.path.join(a.dir, base + ".h5")
        scorer, hkey = original_scorer_and_key(orig_h5)
        scorer = scorer or base

        cols = pd.MultiIndex.from_product([[scorer], bps, ["x", "y", "likelihood"]],
                                          names=["scorer", "bodyparts", "coords"])
        blocks = [np.column_stack([xy[:, c, 0], xy[:, c, 1], pp[:, c]]) for c in conv]
        out = pd.DataFrame(np.hstack(blocks), columns=cols)
        out.to_hdf(out_h5, key=hkey, mode="w")
        if a.csv:
            out.to_csv(out_h5[:-3] + ".csv")
        cov = 100 * np.isfinite(xy[:, conv[0], 0]).mean()
        print(f"  wrote {os.path.basename(out_h5)}  (detection coverage {cov:.1f}%)")

        # Validate against human labels wherever they exist.
        lp = glob.glob(os.path.join(a.project, "labeled-data", video,
                                    "CollectedData_*.h5"))
        if not lp:
            print("  no labels for this video -- not validated")
            continue
        L = top(pd.read_hdf(lp[0]))
        O = top(pd.read_hdf(orig_h5)) if os.path.exists(orig_h5) else None
        N = top(out)
        res = {}
        for bp in bps:
            if bp not in L.columns.get_level_values(0):
                continue
            eo, en = [], []
            for ix in L.index:
                fn = ix[-1] if isinstance(ix, tuple) else os.path.basename(str(ix))
                dg = "".join(ch for ch in fn if ch.isdigit())
                if not dg:
                    continue
                fr = int(dg)
                if fr >= len(N):
                    continue
                lx, ly = float(L.loc[ix, (bp, "x")]), float(L.loc[ix, (bp, "y")])
                if not np.isfinite([lx, ly]).all():
                    continue
                nx, ny = float(N.iloc[fr][(bp, "x")]), float(N.iloc[fr][(bp, "y")])
                if np.isfinite([nx, ny]).all():
                    en.append(np.hypot(nx - lx, ny - ly))
                if O is not None:
                    ox, oy = float(O.iloc[fr][(bp, "x")]), float(O.iloc[fr][(bp, "y")])
                    if np.isfinite([ox, oy]).all():
                        eo.append(np.hypot(ox - lx, oy - ly))
            if en:
                res[bp] = (np.median(eo) if eo else np.nan, np.median(en), len(en))
        if res:
            print("  validation against human labels (median px):")
            print(f"    {'bodypart':14s} {'before':>8s} {'after':>8s} {'n':>5s}")
            for bp, (b, n_, k) in res.items():
                mark = ""
                if np.isfinite(b) and n_ < 0.6 * b:
                    mark = "   <-- repaired"
                elif np.isfinite(b) and n_ > 1.4 * b:
                    mark = "   <-- WORSE, do not use"
                print(f"    {bp:14s} {b:8.2f} {n_:8.2f} {k:5d}{mark}")
                summary.append(dict(video=video, bodypart=bp, before=b, after=n_, n=k))

    if summary:
        S = pd.DataFrame(summary)
        S.to_csv("rebuild_validation.csv", index=False)
        print("\n=== overall median px across validated videos")
        print(S.groupby("bodypart")[["before", "after"]].median().round(2).to_string())
        print("\nwrote rebuild_validation.csv")
    print("\nOriginal h5 files were not modified. Point downstream analysis at the "
          f"'{a.suffix}' files once the validation above looks right.")


if __name__ == "__main__":
    main()
