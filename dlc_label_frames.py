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

STRATA, in the order a frame is charged to them
  dlc_miss_blob_hit  DLC found nothing where blob_tracker.py found a gated
            detection. The animal is demonstrably visible and the network
            demonstrably failed, which is the most informative label there is.
            Needs --track.
  far_from_blob  both trackers fired but disagree by more than --far-px, so one
            of them is wrong. Needs --track.
  dropout_short  brief runs of detector failure (shorter than --long-run). The
            animal is usually mid-stride and the frame is easy to label.
  dropout_long   sustained failures, sampled by SITE rather than by run. These
            concentrate -- on C, 55 of 249 sustained runs sat around one spot
            and carried 14,864 of 42,268 missing frames, the longest run 9,685
            frames -- so --max-per-site keeps one parked animal from eating the
            budget.
  jump      the centre teleports between two valid detections: the detector
            locking onto the cable, a reflection, or the experimenter's hand.
            Threshold is the --jump-pct percentile unless --jump-px is given.
  coverage  k-means over (x, y, t) on valid detections, so the label set spans
            the maze and the session instead of the animal's favourite corner.
            Absorbs whatever budget the failure strata do not spend.

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


def failure_runs(bad):
    """Start, length and mid-frame of every run of detector failure."""
    ch = np.flatnonzero(np.diff(np.r_[0, bad.view(np.int8), 0]))
    st, en = ch[::2], ch[1::2]
    return pd.DataFrame(dict(start=st, length=en - st, mid=(st + en - 1) // 2))


def gap_location(R, x, y):
    """Where each failure run happened, from the flanking valid detections."""
    n = len(x)
    pre = np.clip(R.start.to_numpy() - 1, 0, n - 1)
    post = np.clip(R.start.to_numpy() + R.length.to_numpy(), 0, n - 1)
    ok = (x[pre] > 0) & (x[post] > 0)
    gx = np.where(ok, (x[pre] + x[post]) / 2, np.nan)
    gy = np.where(ok, (y[pre] + y[post]) / 2, np.nan)
    return gx, gy


def _site_capped(R, gx, gy, budget, n_sites, max_per_site, seed):
    """Spread picks over failure SITES, not over runs.

    Sustained failures concentrate: on C, 55 of 249 long runs sat within a few
    tens of pixels of one spot and carried 14,864 of the 42,268 missing frames,
    the longest single run 9,685 frames. Sampling runs in proportion to their
    length would spend most of the budget on one stationary pose, so cluster the
    run locations and take at most `max_per_site` from each.
    """
    if not len(R) or budget <= 0:
        return np.array([], int)
    rng = np.random.default_rng(seed)
    use = np.isfinite(gx) & np.isfinite(gy)
    if use.sum() < 2:
        return R.mid.to_numpy()[:budget]
    from scipy.cluster.vq import kmeans2
    k = int(min(n_sites, use.sum()))
    Z = np.column_stack([gx[use], gy[use]])
    _, lab = kmeans2(Z, k, minit="++", seed=seed)
    mids = R.mid.to_numpy()[use]
    out = []
    for s in range(k):
        m = np.flatnonzero(lab == s)
        if len(m):
            take = min(max_per_site, len(m))
            out.extend(rng.choice(mids[m], take, replace=False))
    out = np.array(out, int)
    if len(out) > budget:
        out = rng.choice(out, budget, replace=False)
    return out


def strata(T, n, a, seed, blob=None):
    """Frame indices by stratum, deduplicated in priority order.

    Priority runs rarest-and-most-informative first, so a frame that qualifies
    as both a tracker disagreement and generic coverage is charged to the
    former. Coverage absorbs whatever budget the failure strata do not spend.
    """
    rng = np.random.default_rng(seed)
    x, y, f = T.x.to_numpy(float), T.y.to_numpy(float), T.frame.to_numpy(int)
    bad = (x == SENTINEL) & (y == SENTINEL)
    ok = ~bad

    picks = {}

    # 1. disagreement with an independent tracker -- the highest-value class.
    #    DLC found nothing where the blob tracker found a gated detection, so
    #    the animal is demonstrably visible and the network demonstrably failed.
    if blob is not None:
        Bq = blob.reindex(T.frame)
        bx, by = Bq.x.to_numpy(float), Bq.y.to_numpy(float)
        # gap-filled rows are the tracker's interpolation, not evidence that the
        # animal was visible -- they must not count as a disagreement
        blob_ok = (np.isfinite(bx) & np.isfinite(by)
                   & ~Bq.interp.to_numpy(bool, na_value=True))
        miss = np.flatnonzero(bad & blob_ok)
        far = np.flatnonzero(ok & blob_ok &
                             (np.hypot(x - bx, y - by) > a.far_px))
        half = int(round(n * a.frac_disagree / 2))
        for name, cand in (("dlc_miss_blob_hit", miss), ("far_from_blob", far)):
            take = min(half, len(cand))
            picks[name] = rng.choice(cand, take, replace=False) if take else np.array([], int)

    # 2. dropout, split by duration. Brief losses happen with the animal
    #    mid-stride and are almost always labelable; sustained ones are a
    #    different phenomenon (occlusion, a parked animal) and need site caps.
    R = failure_runs(bad)
    gx, gy = gap_location(R, x, y)
    short, lng = R.length < a.long_run, R.length >= a.long_run
    n_drop = int(round(n * a.frac_dropout))
    n_long = int(round(n_drop * a.long_share))
    picks["dropout_long"] = _site_capped(R[lng], gx[lng.to_numpy()],
                                         gy[lng.to_numpy()], n_long,
                                         a.n_sites, a.max_per_site, seed)
    take = min(n_drop - len(picks["dropout_long"]), int(short.sum()))
    picks["dropout_short"] = (rng.choice(R.mid.to_numpy()[short.to_numpy()],
                                         take, replace=False)
                              if take > 0 else np.array([], int))

    # 3. jump: the centre teleports between two valid detections
    d = np.full(len(x), np.nan)
    both = ok[1:] & ok[:-1]
    d[1:][both] = np.hypot(np.diff(x)[both], np.diff(y)[both])
    thr = (np.nanpercentile(d, a.jump_pct) if a.jump_px is None else a.jump_px)
    cand = np.flatnonzero(np.nan_to_num(d) > thr)
    take = min(int(round(n * a.frac_jump)), len(cand))
    picks["jump"] = rng.choice(cand, take, replace=False) if take else np.array([], int)

    # 4. coverage: k-means over (x, y, t) on valid detections
    n_cov = n - sum(len(v) for v in picks.values())
    idx_ok = np.flatnonzero(ok)
    if n_cov > 0 and len(idx_ok) > n_cov:
        from scipy.cluster.vq import kmeans2
        Z = np.column_stack([x[idx_ok], y[idx_ok], f[idx_ok].astype(float)])
        Z = (Z - Z.mean(0)) / np.where(Z.std(0) == 0, 1, Z.std(0))
        cent, lab = kmeans2(Z, n_cov, minit="++", seed=seed)
        cov = [idx_ok[m[((Z[m] - cent[k]) ** 2).sum(1).argmin()]]
               for k in range(n_cov)
               for m in [np.flatnonzero(lab == k)] if len(m)]
        picks["coverage"] = np.array(cov, int)
    else:
        picks["coverage"] = idx_ok[:max(n_cov, 0)]

    # a dropout frame has no coordinate of its own -- report the interpolated
    # gap location so the frame can be placed on a map and sanity-checked
    gap = {int(m): (float(ax_), float(ay_))
           for m, ax_, ay_ in zip(R.mid, gx, gy) if np.isfinite(ax_)}

    order = ["dlc_miss_blob_hit", "far_from_blob", "dropout_short",
             "dropout_long", "jump", "coverage"]
    out, seen = [], set()
    for name in order:
        for r in np.atleast_1d(picks.get(name, np.array([], int))):
            r = int(r)
            if f[r] in seen:
                continue
            seen.add(f[r])
            px, py = (x[r], y[r])
            est = px == SENTINEL
            if est:
                px, py = gap.get(r, (np.nan, np.nan))
            out.append((int(f[r]), name, px, py, est))
    S = pd.DataFrame(out, columns=["frame", "stratum", "x", "y",
                                   "x_is_interpolated"])
    S["thr_jump_px"] = round(float(thr), 1)
    return S.sort_values("frame").reset_index(drop=True), R, (gx, gy)


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


COLOURS = {"coverage": "#9aa3ad", "jump": "#d1682a",
           "dropout_short": "#7a3e9d", "dropout_long": "#c23b8a",
           "dlc_miss_blob_hit": "#1f6fb4", "far_from_blob": "#2e8b57"}


def qc_figure(T, S, R, gxy, long_run, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gx, gy = gxy
    ok = ~((T.x == SENTINEL) & (T.y == SENTINEL))
    sust = (R.length >= long_run).to_numpy()

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))

    ax[0].plot(T.x[ok], T.y[ok], ".", ms=0.3, color="#e2e5e9", rasterized=True)
    for st in S.stratum.unique():
        s = S[S.stratum == st]
        mk = "s" if s.x_is_interpolated.any() else "o"
        ax[0].plot(s.x, s.y, mk, ms=3.2, mfc="none" if mk == "s" else None,
                   mew=0.9, ls="none", color=COLOURS.get(st, "k"),
                   label=f"{st} ({len(s)})")
    ax[0].invert_yaxis(); ax[0].set_aspect("equal")
    ax[0].legend(frameon=False, fontsize=6.5, loc="center", handletextpad=0.4)
    ax[0].set_title("frames picked, by stratum\n"
                    "open squares: position interpolated across the gap",
                    fontsize=8, loc="left")

    ax[1].plot(T.x[ok], T.y[ok], ".", ms=0.3, color="#e2e5e9", rasterized=True)
    sz = np.clip(R.length.to_numpy()[sust] / 20, 4, 200)
    ax[1].scatter(gx[sust], gy[sust], s=sz, color="#c23b8a", alpha=0.55,
                  edgecolor="none", label=f"sustained ({int(sust.sum())} runs)")
    ax[1].scatter(gx[~sust], gy[~sust], s=2, color="#7a3e9d", alpha=0.3,
                  edgecolor="none", label=f"brief ({int((~sust).sum())} runs)")
    ax[1].invert_yaxis(); ax[1].set_aspect("equal")
    ax[1].legend(frameon=False, fontsize=6.5, loc="center")
    ax[1].set_title("where tracking fails (area = run length)", fontsize=9,
                    loc="left")

    L = R.length.to_numpy()
    ax[2].hist(L, bins=np.logspace(0, np.log10(max(L.max(), 10)), 40),
               color="#7a3e9d")
    ax[2].axvline(long_run, color="#c23b8a", lw=1, ls="--")
    ax[2].text(long_run * 1.15, ax[2].get_ylim()[1] * 0.9,
               f"sustained\n>= {long_run} fr", fontsize=6.5, color="#c23b8a",
               va="top")
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel("failure run length (frames)")
    ax[2].set_ylabel("runs")
    ax[2].set_title("two failure modes, not one", fontsize=9, loc="left")

    for b in ax[:2]:
        b.set_xticks([]); b.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=200)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--h5", help="DLC .h5 from analyze_videos")
    src.add_argument("--csv", help="DLC .csv from analyze_videos")
    p.add_argument("--video", help="RAW video; omit with --no-images")
    p.add_argument("--outdir", help="labeled-data/<stem> of the labelling project")
    p.add_argument("--manifest", required=True)
    p.add_argument("--fig", default=None)
    p.add_argument("--bodypart", default="mouse_center")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--track", default=None,
                   help="blob_tracker.py post output; enables the disagreement "
                        "strata, which are the most informative frames there are")
    p.add_argument("--jump-px", type=float, default=None,
                   help="displacement between consecutive frames that counts as "
                        "a failure; default is the --jump-pct percentile")
    p.add_argument("--jump-pct", type=float, default=99.9,
                   help="data-driven jump threshold, used when --jump-px is unset")
    p.add_argument("--far-px", type=float, default=40.0,
                   help="disagreement with the blob tracker that counts as one "
                        "of them being wrong")
    p.add_argument("--long-run", type=int, default=30,
                   help="failure runs this long or longer are 'sustained' and "
                        "are sampled by site rather than by run")
    p.add_argument("--long-share", type=float, default=0.4,
                   help="share of the dropout budget spent on sustained runs")
    p.add_argument("--n-sites", type=int, default=12,
                   help="k-means sites over sustained-failure locations")
    p.add_argument("--max-per-site", type=int, default=3,
                   help="cap per site, so one parked animal cannot dominate")
    p.add_argument("--frac-dropout", type=float, default=0.30)
    p.add_argument("--frac-jump", type=float, default=0.10)
    p.add_argument("--frac-disagree", type=float, default=0.20)
    p.add_argument("--audit", default=None,
                   help="write the failure-run table (start, length, location) "
                        "here; useful even when you are not selecting frames")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-images", action="store_true",
                   help="write the manifest only")
    p.add_argument("--allow-labeled", action="store_true")
    a = p.parse_args()
    if not (a.h5 or a.csv or a.track):
        raise SystemExit("give --h5/--csv (DLC output), or --track alone for a "
                         "video that has never been analysed")

    blob = None
    if a.track:
        B = pd.read_csv(a.track)
        if "frame" not in B.columns:
            raise SystemExit(f"{a.track} has no 'frame' column")
        missing = {"x", "y"} - set(B.columns)
        if missing:
            raise SystemExit(f"{a.track} is missing columns: {sorted(missing)}")
        keep = ["x", "y"] + (["interp"] if "interp" in B.columns else [])
        blob = B.set_index("frame")[keep]
        if "interp" not in blob.columns:
            print("WARNING: no 'interp' column -- cannot tell real blob "
                  "detections from gap-filled ones; treating all as real")
            blob["interp"] = False
        real = np.isfinite(blob.x) & ~blob.interp.astype(bool)
        print(f"blob track: {len(blob)} rows, {int(real.sum())} real detections, "
              f"{int(blob.interp.astype(bool).sum())} gap-filled (excluded)")

    if a.h5 or a.csv:
        T = read_track(a.h5, a.csv, a.bodypart)
        bad = int(((T.x == SENTINEL) & (T.y == SENTINEL)).sum())
        print(f"{len(T)} frames, {bad} with no detection "
              f"({100 * bad / len(T):.1f}%), likelihood median "
              f"{T.likelihood.median():.4f} max {T.likelihood.max():.3f}")
    else:
        # Without DLC output, use real blob detections as the primary track.
        # Interpolated rows remain failures rather than pseudo-observations.
        print("no DLC output given -- blob-only mode, disagreement strata off")
        T = pd.DataFrame({
            "frame": blob.index.to_numpy(int),
            "x": np.where(real, blob.x, SENTINEL),
            "y": np.where(real, blob.y, SENTINEL),
            "likelihood": np.nan,
        })
        blob = None
        bad = int((T.x == SENTINEL).sum())
        print(f"{len(T)} frames, {bad} the blob tracker lost "
              f"({100 * bad / len(T):.1f}%)")

    S, R, (gx, gy) = strata(T, a.n, a, a.seed, blob)
    brief, sust = R[R.length < a.long_run], R[R.length >= a.long_run]
    print(f"failure runs: {len(brief)} brief (<{a.long_run} fr, "
          f"{int(brief.length.sum())} frames), {len(sust)} sustained "
          f"({int(sust.length.sum())} frames, longest {int(R.length.max())})")
    print("picked " + ", ".join(f"{k} {v}" for k, v in
                                S.stratum.value_counts().items()))
    if a.audit:
        R.assign(gap_x=gx, gap_y=gy,
                 kind=np.where(R.length >= a.long_run, "sustained", "brief")
                 ).to_csv(a.audit, index=False)
        print(f"wrote {a.audit}")

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
        qc_figure(T, S, R, (gx, gy), a.long_run, a.fig)
        print(f"wrote {a.fig}")


if __name__ == "__main__":
    main()
