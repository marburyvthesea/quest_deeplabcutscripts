"""Does analyze_videos agree with evaluate_network on the same frames?

The Oct-20 fine-tune is a SuperAnimal memory-replay run. train.txt shows the
pose head emits 27 heatmaps -- the full superanimal_topviewmouse keypoint set
-- and the five project bodyparts are channels [0, 1, 2, 9, 13] via
conversion_array, with body_center at channel 9. Every snapshot confirms the
27-wide head, yet the analysis h5 carries exactly 5 correctly-named bodypart
columns. So a channel selection happens between head and file.

If that selection is right in the evaluation path but wrong in the video path,
predictions would land on the animal but displaced along the body axis -- the
neighbouring superanimal channels are the other spine points -- while DLC's own
RMSE stayed low. That is exactly the pattern diag_axis.py measured: offset
tightly constrained perpendicular to the body axis (IQR 6-14 px) and scattered
along it by nearly a body length (IQR 30-33 px).

evaluate_network had never actually been run on this project: the directory
evaluation-results-pytorch/iteration-0/<trainset>/ is empty, and the 6.16 px
test RMSE in the benchmark report came from the in-training eval loop in
learning_stats.csv, not from a held-out evaluation pass.

Reading the result:

  eval_err ~= vid_err   both paths agree -- the 18 px is genuine pose error
                        and the channel hypothesis is dead
  eval_err <  vid_err   the video path is wrong; eval_vs_vid tells you by how
                        much, and a per-bodypart breakdown says whether it is
                        body_center alone (channel 9 specifically) or all five
                        (a shifted/ignored conversion_array)

Usage:
  python diag_eval_vs_video.py --config .../config.yaml
  python diag_eval_vs_video.py --config ... --skip-eval   # reuse existing preds
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

D_DEFAULT = "/scratch/jma819/T_maze_recordings/behaviorVIdeos/dlc_out_finetuned"


def parse_key(ix):
    """('labeled-data', 'C', 'img0012345.png') or a path string -> ('C', 12345)."""
    s = os.path.join(*ix) if isinstance(ix, tuple) else str(ix)
    m = re.search(r"labeled-data[\\/]([^\\/]+)[\\/]img(\d+)\.png", s)
    if m:
        return m.group(1), int(m.group(2))
    m = re.search(r"([^\\/]+)[\\/]img(\d+)\.png", s)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def top_scorer(df):
    """Drop the scorer level, whatever it is called."""
    return df[df.columns.get_level_values(0)[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dir", default=D_DEFAULT, help="analyze_videos h5 directory")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--shuffle", type=int, default=1)
    a = ap.parse_args()

    proj = os.path.dirname(os.path.abspath(a.config))

    if not a.skip_eval:
        import deeplabcut
        print("running evaluate_network -- this writes the per-image predictions "
              "that the project is missing", flush=True)
        deeplabcut.evaluate_network(a.config, Shuffles=[a.shuffle], plotting=False)

    E = sorted(glob.glob(proj + "/evaluation-results*/**/*.h5", recursive=True))
    print("evaluation prediction files:", [os.path.basename(e) for e in E])
    if not E:
        raise SystemExit("evaluate_network produced no h5 -- cannot compare")
    # Prefer a file naming the snapshot the videos were analysed with.
    pref = [e for e in E if "snapshot" in os.path.basename(e)]
    ev = pd.read_hdf(pref[0] if pref else E[0])
    print(f"using {os.path.basename(pref[0] if pref else E[0])}  shape={ev.shape}")

    keys = [parse_key(i) for i in ev.index]
    videos = sorted({v for v, f in keys if v is not None})
    print("videos represented in the labelled set:", videos)

    EV = top_scorer(ev)
    parts = list(dict.fromkeys(EV.columns.get_level_values(0)))
    print("bodyparts:", parts)

    rows = []
    for v in videos:
        lab_p = os.path.join(proj, "labeled-data", v, "CollectedData_jjm.h5")
        vid_p = sorted(glob.glob(os.path.join(a.dir, v + "DLC*.h5")))
        if not os.path.exists(lab_p) or not vid_p:
            print(f"  skip {v}: labels={os.path.exists(lab_p)} video_h5={bool(vid_p)}")
            continue
        LAB = top_scorer(pd.read_hdf(lab_p))
        VID = top_scorer(pd.read_hdf(vid_p[0]))
        lab_ix = {parse_key(i): i for i in LAB.index}
        n_v = len(VID)

        for (vv, fr), ix in zip(keys, ev.index):
            if vv != v or (vv, fr) not in lab_ix or fr >= n_v:
                continue
            li = lab_ix[(vv, fr)]
            for bp in parts:
                if bp not in LAB.columns.get_level_values(0):
                    continue
                lx, ly = LAB.loc[li, (bp, "x")], LAB.loc[li, (bp, "y")]
                sx, sy = EV.loc[ix, (bp, "x")], EV.loc[ix, (bp, "y")]
                vx, vy = VID.iloc[fr][(bp, "x")], VID.iloc[fr][(bp, "y")]
                vals = np.array([lx, ly, sx, sy, vx, vy], dtype=float)
                if not np.isfinite(vals).all():
                    continue
                rows.append(dict(
                    video=v, frame=fr, bodypart=bp,
                    eval_err=float(np.hypot(sx - lx, sy - ly)),
                    vid_err=float(np.hypot(vx - lx, vy - ly)),
                    eval_vs_vid=float(np.hypot(sx - vx, sy - vy)),
                ))

    R = pd.DataFrame(rows)
    if R.empty:
        raise SystemExit("no frames matched between labels, evaluation and video h5")
    R.to_csv("eval_vs_video.csv", index=False)

    cols = ["eval_err", "vid_err", "eval_vs_vid"]
    print(f"\nn = {len(R)} (frame x bodypart) comparisons\n")
    print("--- median px by bodypart")
    print(R.groupby("bodypart")[cols].median().round(2).to_string())
    print("\n--- median px by video")
    print(R.groupby("video")[cols].median().round(2).to_string())
    print("\n--- body_center only, by video")
    bc = R[R.bodypart == "body_center"]
    if len(bc):
        print(bc.groupby("video")[cols].median().round(2).to_string())

    m = R[cols].median()
    print("\nOVERALL  eval_err %.2f px | vid_err %.2f px | eval_vs_vid %.2f px"
          % (m.eval_err, m.vid_err, m.eval_vs_vid))
    ratio = m.vid_err / max(m.eval_err, 1e-9)
    if m.eval_vs_vid < 3:
        print("VERDICT: the two paths agree -- the error is genuine pose error, "
              "not a channel-selection bug.")
    elif ratio > 2:
        print("VERDICT: the video path is %.1fx worse than evaluation on the same "
              "frames -- the fault is in analyze_videos, not the network." % ratio)
    else:
        print("VERDICT: paths differ but not dramatically; inspect "
              "eval_vs_video.csv per bodypart before concluding.")
    print("\nwrote eval_vs_video.csv")


if __name__ == "__main__":
    main()
