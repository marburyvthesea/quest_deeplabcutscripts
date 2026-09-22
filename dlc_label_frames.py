#!/usr/bin/env python
"""
Pick frames for HUMAN labelling from an analysed video, stratified so that the
pose head has something to learn from.

WHY NOT deeplabcut.extract_outlier_frames
-----------------------------------------
The obvious tool does not work on this model. The TmazeBlob pose head never
trained -- 26 of its 27 keypoint channels had no labels, so the WeightedMSE
gradient never moved and every likelihood in the output sits near 0.003
(C: median 0.0028, 99th pct 0.0068, max 0.41; see dlc_analyze_finetuned.py).
Consequently:

  outlieralgorithm="uncertain"  flags all 251,334 frames at any sane p_bound
  outlieralgorithm="fitting"    fits an ARIMA to a flat noise floor
  outlieralgorithm="jump"       does work, because it keys on position

So selection here keys on the two parts of the output that DO carry
information: the detector's verdict (x == -1 when it found no animal) and
mouse_center's position.

STRATA
  coverage  k-means over (x, y, t) on validly detected frames, so the label set
            spans the maze and the session instead of the animal's favourite
            corner. This is the same idea as DLC's kmeans extraction, run on
            tracked position rather than pixels because position is what the
            downstream analysis uses.
  dropout   frames drawn from runs of detector failure (x == y == -1). These
            were 17% of C and are exactly the frames the network needs.
            Sampled one per run so a single long failure cannot dominate.
  jump      frames where mouse_center moves more than --jump-px between
            consecutive samples: the detector locking onto the cable, a
            reflection, or the experimenter's hand.

Images come from the RAW video. The script refuses a DLC-labeled overlay --
extracting from one bakes the old markers into the training images, and the
network will happily learn to find the marker instead of the mouse.

  python dlc_label_frames.py \
      --h5 /scratch/jma819/T_maze_recordings/behaviorVIdeos/dlc_out_finetuned/\
CDLC_HrnetW32_TmazeBlob2026-09-17shuffle1_detector_best-100_snapshot_best-10.h5 \
      --video /scratch/jma819/T_maze_recordings/behaviorVIdeos/C.avi \
      --outdir /projects/b1118/dlc_analysis/TmazeParts-label/labeled-data/C \
      --n 200 --manifest C_labelframes.csv --fig C_labelframes.png
"""
import argparse
import os

import numpy as np
import pandas as pd


SENTINEL = -1.0


def read_track(h5=None, csv=None, bodypart="mouse_center"):
    """Frame-indexed x, y, likelihood for one bodypart of a DLC output."""
    if h5:
        df = pd.read_hdf(h5)
    else:
        df = pd.read_csv(csv, header=[0, 1, 2], index_col=0)
    if df.columns.nlevels == 4:                      # multi-animal
        df = df.droplevel("individuals", axis=1)
    df = df.droplevel("scorer", axis=1)
    parts = list(dict.fromkeys(df.columns.get_level_values("bodyparts")))
    if bodypart not in parts:
        raise SystemExit(
            f"--bodypart {bodypart!r} not in the file; it has {len(parts)}: "
            f"{parts[:6]}{' ...' if len(parts) > 6 else ''}"
        )
    T = df[bodypart].copy()
    T.columns = [str(c) for c in T.columns]
    T["frame"] = np.asarray(df.index, dtype=int)
    return T.reset_index(drop=True)


def strata(T, n, jump_px, frac_dropout, frac_jump, seed):
    """Frame indices in three strata, deduplicated, coverage taking the rest."""
    rng = np.random.default_rng(seed)
    x, y, f = T.x.to_numpy(float), T.y.to_numpy(float), T.frame.to_numpy(int)
    bad = (x == SENTINEL) & (y == SENTINEL)

    # dropout: one frame from the middle of each run of detector failure
    runs, i = [], 0
    while i < len(bad):
        if bad[i]:
            j = i
            while j + 1 < len(bad) and bad[j + 1]:
                j += 1
            runs.append((i + j) // 2)
            i = j + 1
        else:
            i += 1
    n_drop = min(int(round(n * frac_dropout)), len(runs))
    pick_drop = rng.choice(runs, n_drop, replace=False) if n_drop else np.array([], int)

    # jump: large frame-to-frame displacement between two valid detections
    ok = ~bad
    d = np.full(len(x), np.nan)
    both = ok[1:] & ok[:-1]
    d[1:][both] = np.hypot(np.diff(x)[both], np.diff(y)[both])
    cand = np.flatnonzero(np.nan_to_num(d) > jump_px)
    n_jump = min(int(round(n * frac_jump)), len(cand))
    pick_jump = rng.choice(cand, n_jump, replace=False) if n_jump else np.array([], int)

    # coverage: k-means over (x, y, t), z-scored, on validly detected frames
    n_cov = n - len(pick_drop) - len(pick_jump)
    idx_ok = np.flatnonzero(ok)
    if n_cov > 0 and len(idx_ok) > n_cov:
        from scipy.cluster.vq import kmeans2
        Z = np.column_stack([x[idx_ok], y[idx_ok], f[idx_ok].astype(float)])
        Z = (Z - Z.mean(0)) / np.where(Z.std(0) == 0, 1, Z.std(0))
        cent, lab = kmeans2(Z, n_cov, minit="++", seed=seed)
        pick_cov = []
        for k in range(n_cov):
            m = np.flatnonzero(lab == k)
            if len(m):
                dd = ((Z[m] - cent[k]) ** 2).sum(1)
                pick_cov.append(idx_ok[m[dd.argmin()]])
        pick_cov = np.array(pick_cov, int)
    else:
        pick_cov = idx_ok[:max(n_cov, 0)]

    out, seen = [], set()
    for rows, name in ((pick_drop, "dropout"), (pick_jump, "jump"),
                       (pick_cov, "coverage")):
        for r in np.atleast_1d(rows):
            if int(f[r]) not in seen:
                seen.add(int(f[r]))
                out.append((int(f[r]), name, x[r], y[r]))
    S = pd.DataFrame(out, columns=["frame", "stratum", "x", "y"])
    return S.sort_values("frame").reset_index(drop=True)


def write_images(S, video, outdir, allow_labeled=False):
    import cv2
    base = os.path.basename(video).lower()
    if not allow_labeled and ("labeled" in base or "_p60" in base):
        raise SystemExit(
            f"{base} looks like a DLC overlay. Extract from the raw video or the "
            "old markers become part of the training images (--allow-labeled to "
            "override)."
        )
    os.makedirs(outdir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if S.frame.max() >= n_vid:
        raise SystemExit(
            f"tracking indexes frame {S.frame.max()} but {base} has {n_vid}. "
            "The h5 and the video are not the same recording."
        )
    written = []
    for fr in S.frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
        ok, im = cap.read()
        if ok:
            nm = f"img{int(fr):07d}.png"
            cv2.imwrite(os.path.join(outdir, nm), im)
            written.append(nm)
        else:
            written.append("")
    cap.release()
    return written


def qc_figure(T, S, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = ~((T.x == SENTINEL) & (T.y == SENTINEL))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot(T.x[ok], T.y[ok], ".", ms=0.4, color="#c8ccd1")
    for st, c in (("coverage", "#1f6fb4"), ("jump", "#d1682a"),
                  ("dropout", "#7a3e9d")):
        s = S[S.stratum == st]
        ax[0].plot(s.x.where(s.x > 0), s.y.where(s.y > 0), "o", ms=3,
                   color=c, label=f"{st} ({len(s)})")
    ax[0].invert_yaxis(); ax[0].set_aspect("equal")
    ax[0].legend(frameon=False, fontsize=7)
    ax[0].set_title("where the label frames come from", fontsize=9)
    ax[1].hist([S[S.stratum == s].frame for s in ("coverage", "jump", "dropout")],
               bins=40, stacked=True,
               color=["#1f6fb4", "#d1682a", "#7a3e9d"])
    ax[1].set_xlabel("video frame"); ax[1].set_ylabel("frames picked")
    ax[1].set_title("spread over the session", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--h5", help="DLC .h5 from analyze_videos")
    src.add_argument("--csv", help="DLC .csv from analyze_videos")
    p.add_argument("--video", help="RAW video; omit with --no-images")
    p.add_argument("--outdir", help="labeled-data/<stem> of the labelling project")
    p.add_argument("--manifest", required=True)
    p.add_argument("--fig", default=None)
    p.add_argument("--bodypart", default="mouse_center")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--jump-px", type=float, default=60.0,
                   help="displacement between consecutive frames that counts "
                        "as a tracking failure")
    p.add_argument("--frac-dropout", type=float, default=0.25)
    p.add_argument("--frac-jump", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-images", action="store_true",
                   help="write the manifest only")
    p.add_argument("--allow-labeled", action="store_true")
    a = p.parse_args()

    T = read_track(a.h5, a.csv, a.bodypart)
    bad = int(((T.x == SENTINEL) & (T.y == SENTINEL)).sum())
    print(f"{len(T)} frames, {bad} with no detection ({100 * bad / len(T):.1f}%), "
          f"likelihood median {T.likelihood.median():.4f} max {T.likelihood.max():.3f}")
    S = strata(T, a.n, a.jump_px, a.frac_dropout, a.frac_jump, a.seed)
    print("picked " + ", ".join(f"{k} {v}" for k, v in
                                S.stratum.value_counts().items()))

    S["image"] = [f"img{int(f):07d}.png" for f in S.frame]
    if not a.no_images:
        if not (a.video and a.outdir):
            raise SystemExit("--video and --outdir are required without --no-images")
        got = write_images(S, a.video, a.outdir, a.allow_labeled)
        S["image"] = got
        n_ok = int((S.image != "").sum())
        print(f"wrote {n_ok} images to {a.outdir}/")
        if n_ok < len(S):
            print(f"WARNING {len(S) - n_ok} frames would not decode")
    S.to_csv(a.manifest, index=False)
    print(f"wrote {a.manifest}")
    if a.fig:
        qc_figure(T, S, a.fig)
        print(f"wrote {a.fig}")


if __name__ == "__main__":
    main()
