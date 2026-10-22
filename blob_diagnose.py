#!/usr/bin/env python
"""Diagnose blob_tracker failures against the video itself.

`blob_tracker.py audit` splits the misses by cause from the track CSV alone.
That tells you which question to ask; these subcommands answer it by going back
to the frames.

    frames  contact sheet of failure frames with the last known position marked.
            The eye settles in seconds what no statistic will: cable over the
            body, animal off the masked surface, animal merged with a dark
            fixture, or animal plainly visible -- which means the gates failed.
    probe   re-runs candidates() at each miss and measures how far the nearest
            candidate sat from the last accepted position. Answers the
            association question numerically: what --max-step would recover
            these frames, and at what cost.
    sweep   re-runs candidates() over a (--dark-thr x --min-depth) grid on miss
            frames only, and on a control sample of tracked frames. Answers the
            segmentation question: which loosening recovers misses without
            flooding good frames with spurious blobs.
    clips   mp4 around the worst runs with every candidate drawn, accepted or
            not, plus the surface boundary.

Everything imports blob_tracker and calls its own candidates()/pick(), so a
diagnostic result is a statement about the pipeline that ran and not about a
reimplementation of it. Reuse the cached background from `blob_tracker.py mask`
(--bg <out>_bg.npy) or these will each spend a pass rebuilding it.
"""
import argparse
import json
import os

import sys

import cv2
import numpy as np
import pandas as pd

# import the tracker as a sibling module however this script is invoked -- the
# diagnostics are only meaningful if they call the same candidates()/pick() the
# track ran, so this import is load-bearing rather than a convenience.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blob_tracker as bt  # noqa: E402

CAUSE = {"segmentation": 0, "association": 1}


# ---------------------------------------------------------------- shared

def load_failures(track, audit=None):
    """Miss frames and their cause, with the last real detection before each."""
    T = pd.read_csv(track)
    status = T.status.to_numpy() if "status" in T.columns else None
    miss = (status == "miss") if status is not None else ~np.isfinite(T.x.to_numpy(float))
    nc = T.n_cand.to_numpy(float) if "n_cand" in T.columns else np.zeros(len(T))
    real = ~miss
    if "interp" in T.columns:
        real &= ~T.interp.to_numpy(bool)
    x, y = T.x.to_numpy(float), T.y.to_numpy(float)
    frame = T.frame.to_numpy(int) if "frame" in T.columns else np.arange(len(T))

    runs = bt.runs_of(miss)
    rows = []
    for s0, s1 in runs:
        before = np.flatnonzero(real[:s0])
        prev = (x[before[-1]], y[before[-1]]) if before.size else (np.nan, np.nan)
        gap0 = s0 - before[-1] - 1 if before.size else 0
        cause = "association" if nc[s0:s1].max(initial=0) > 0 else "segmentation"
        for k in range(s0, s1):
            rows.append(dict(frame=int(frame[k]), cause=cause, run_start=int(s0),
                             run_len=int(s1 - s0), pos_in_run=int(k - s0),
                             n_cand=float(nc[k]), prev_x=prev[0], prev_y=prev[1],
                             gap=int(gap0 + k - s0)))
    return T, pd.DataFrame(rows)


def sample_runs(F, n, per_run=1, seed=0):
    """Spread the budget across runs rather than across frames.

    Consecutive frames of one dropout show the same animal in the same place, so
    sampling frames uniformly buys repeats. One frame per run, runs drawn in
    proportion to how many frames their cause lost.
    """
    if F.empty:
        return F
    rng = np.random.default_rng(seed)
    groups = dict(tuple(F.groupby("cause")))
    avail = {c: g.run_start.nunique() for c, g in groups.items()}

    # Budget per cause, NOT weighted by frames lost: association typically loses
    # an order of magnitude fewer frames than segmentation but is the cause worth
    # looking at, and a frame-weighted split starves it. Start even, then hand
    # any budget a cause cannot spend (fewer runs than slots) to the others, so
    # the sheet actually fills to --n.
    quota = {c: 0 for c in groups}
    left = max(1, n // max(1, per_run))
    while left > 0 and any(quota[c] < avail[c] for c in groups):
        hungry = [c for c in groups if quota[c] < avail[c]]
        for c in sorted(hungry, key=lambda c: (quota[c], -len(groups[c]))):
            if left == 0:
                break
            quota[c] += 1
            left -= 1

    out = []
    for cause, g in groups.items():
        runs = g.run_start.unique()
        take = rng.choice(runs, min(len(runs), quota[cause]), replace=False)
        for r in take:
            gr = g[g.run_start == r]
            out.append(gr.iloc[[min(len(gr) - 1, len(gr) // 2)] * per_run])
    return pd.concat(out).sort_values("frame").head(n)


def read_frames(video, frames):
    """Decode a scattered set of frames, sorted so each seek moves forward."""
    cap, N = bt.open_video(video)
    got = {}
    for f in sorted(set(int(v) for v in frames)):
        if f >= N:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if ok:
            got[f] = im
    cap.release()
    return got, N


def get_bg_mask(a):
    g = json.load(open(a.geom))
    if a.bg and os.path.exists(a.bg):
        bg = np.load(a.bg)
    else:
        cap, N = bt.open_video(a.video)
        bg = bt.background(cap, N, a.bg_stride)
        cap.release()
    return bg, bt.surface_mask(bg, g, a.sat_thr, a.erode), g


# ---------------------------------------------------------------- frames

SHORT = {"segmentation": "seg", "association": "assoc"}


def cmd_frames(a):
    T, F = load_failures(a.track, a.audit)
    if F.empty:
        print("no misses in that track -- nothing to show")
        return
    S = sample_runs(F, a.n, seed=a.seed)
    got, N = read_frames(a.video, S.frame)
    S = S[S.frame.isin(got)]
    print(f"{len(F)} miss frames in {F.run_start.nunique()} runs; "
          f"showing {len(S)} ({S.cause.value_counts().to_dict()})")

    cols = max(1, min(a.cols, len(S)))
    rows = int(np.ceil(len(S) / cols))
    h, w = next(iter(got.values())).shape[:2]
    sc = a.tile / w
    th, tw = int(h * sc), int(w * sc)
    sheet = np.full((rows * (th + 18), cols * tw, 3), 255, np.uint8)
    for i, (_, r) in enumerate(S.reset_index().iterrows()):
        im = cv2.resize(got[int(r.frame)], (tw, th))
        if np.isfinite(r.prev_x):
            cv2.circle(im, (int(r.prev_x * sc), int(r.prev_y * sc)),
                       max(3, int(18 * sc)), (0, 200, 255), 1)
        ry, cx = divmod(i, cols)
        y0 = ry * (th + 18)
        sheet[y0:y0 + th, cx * tw:(cx + 1) * tw] = im
        cv2.putText(sheet, f"f{int(r.frame)}  {SHORT[r.cause]}  {int(r.run_len)}fr",
                    (cx * tw + 2, y0 + th + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(a.out, sheet)
    print(f"wrote {a.out}  (circle = last accepted position; if the animal is "
          f"inside it and clearly visible, the gates rejected a real blob)")
    if a.outdir:
        os.makedirs(a.outdir, exist_ok=True)
        for f in S.frame:
            cv2.imwrite(os.path.join(a.outdir, f"miss{int(f):07d}.png"), got[int(f)])
        print(f"wrote {len(S)} frames to {a.outdir}")


# ---------------------------------------------------------------- probe

def cmd_probe(a):
    T, F = load_failures(a.track, a.audit)
    F = F[np.isfinite(F.prev_x)]
    if F.empty:
        print("no misses with a preceding detection -- nothing to probe")
        return
    S = sample_runs(F, a.n, seed=a.seed)
    bg, mask, _ = get_bg_mask(a)
    got, N = read_frames(a.video, S.frame)

    rec = []
    for _, r in S.iterrows():
        im = got.get(int(r.frame))
        if im is None:
            continue
        grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        cands, _, _ = bt.candidates(grey, bg, mask, a.dark_thr,
                                    a.area[0], a.area[1], a.min_depth)
        allow = a.max_step * max(1, r.gap + 1)
        if cands:
            d = np.array([np.hypot(c["x"] - r.prev_x, c["y"] - r.prev_y) for c in cands])
            j = int(np.argmin(d))
            best = cands[int(np.argmax([c["score"] for c in cands]))]
            rec.append(dict(frame=int(r.frame), cause=r.cause, gap=int(r.gap),
                            n_cand=len(cands), nearest_px=float(d[j]), allow_px=allow,
                            nearest_score=cands[j]["score"], top_score=best["score"],
                            nearest_area=cands[j]["area"], nearest_depth=cands[j]["depth"],
                            nearest_elong=cands[j]["elong"]))
        else:
            rec.append(dict(frame=int(r.frame), cause=r.cause, gap=int(r.gap),
                            n_cand=0, nearest_px=np.nan, allow_px=allow,
                            nearest_score=np.nan, top_score=np.nan,
                            nearest_area=np.nan, nearest_depth=np.nan,
                            nearest_elong=np.nan))
    R = pd.DataFrame(rec)
    have = R[R.n_cand > 0]
    print(f"probed {len(R)} miss frames: {len(have)} had >=1 candidate on re-run, "
          f"{len(R) - len(have)} had none")
    if len(have):
        over = have[have.nearest_px > have.allow_px]
        print(f"  nearest candidate: median {have.nearest_px.median():.1f} px, "
              f"gate median {have.allow_px.median():.1f} px")
        print(f"  {len(over)} of {len(have)} were outside the gate "
              f"({100 * len(over) / len(have):.0f}%) -- these are pure association loss")
        for mult in (1.5, 2.0, 3.0):
            rec_n = (have.nearest_px <= have.allow_px * mult).mean()
            print(f"  --max-step x{mult:.1f} would admit {100 * rec_n:.0f}% of them")
    R.to_csv(a.out, index=False)
    print(f"wrote {a.out}")

    if a.fig and len(have):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))
        axs[0].scatter(have.allow_px, have.nearest_px, s=10, alpha=0.5)
        lim = float(np.nanmax([have.allow_px.max(), have.nearest_px.max()])) * 1.05
        axs[0].plot([0, lim], [0, lim], "k--", lw=1)
        axs[0].set(xlabel="gate, max_step x (gap+1)  (px)",
                   ylabel="nearest candidate (px)",
                   title="above the line = rejected by the gate")
        axs[1].hist(have.nearest_px / have.allow_px, bins=40, color="0.4")
        axs[1].axvline(1.0, color="crimson", lw=1.4)
        axs[1].set(xlabel="nearest candidate / gate", ylabel="frames",
                   title="how far past the gate")
        fig.tight_layout()
        fig.savefig(a.fig, dpi=150)
        print(f"wrote {a.fig}")


# ---------------------------------------------------------------- sweep

def cmd_sweep(a):
    T, F = load_failures(a.track, a.audit)
    seg = F[F.cause == "segmentation"]
    if seg.empty:
        print("no segmentation-class misses -- loosening thresholds has nothing to recover")
        return
    S = sample_runs(seg, a.n, seed=a.seed)
    real = T.status.to_numpy() == "ok" if "status" in T.columns else np.isfinite(T.x)
    ctrl_idx = np.flatnonzero(real)
    rng = np.random.default_rng(a.seed)
    ctrl = rng.choice(ctrl_idx, min(a.n_control, ctrl_idx.size), replace=False)
    ctrl_frames = (T.frame.to_numpy(int)[ctrl] if "frame" in T.columns else ctrl)

    bg, mask, _ = get_bg_mask(a)
    got, N = read_frames(a.video, list(S.frame) + list(ctrl_frames))
    grid = [(d, m) for d in a.dark_grid for m in a.depth_grid]
    print(f"{len(S)} miss frames, {len(ctrl_frames)} control frames, "
          f"{len(grid)} grid points")

    out = []
    for dark, mind in grid:
        rec, ncand_miss, ncand_ctrl = 0, [], []
        for f in S.frame:
            im = got.get(int(f))
            if im is None:
                continue
            grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
            c, _, _ = bt.candidates(grey, bg, mask, dark, a.area[0], a.area[1], mind)
            rec += len(c) > 0
            ncand_miss.append(len(c))
        for f in ctrl_frames:
            im = got.get(int(f))
            if im is None:
                continue
            grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
            c, _, _ = bt.candidates(grey, bg, mask, dark, a.area[0], a.area[1], mind)
            ncand_ctrl.append(len(c))
        out.append(dict(dark_thr=dark, min_depth=mind,
                        recovered=rec / max(len(ncand_miss), 1),
                        cand_on_miss=float(np.mean(ncand_miss or [0])),
                        cand_on_ok=float(np.mean(ncand_ctrl or [0]))))
        print(f"  dark_thr={dark:5.0f} min_depth={mind:5.1f}  "
              f"recovered {100 * out[-1]['recovered']:5.1f}%  "
              f"candidates/frame on tracked {out[-1]['cand_on_ok']:.2f}")
    G = pd.DataFrame(out)
    G.to_csv(a.out, index=False)
    print(f"wrote {a.out}")
    base = G[(G.dark_thr == a.dark_thr) & (G.min_depth == a.min_depth)]
    if len(base):
        print(f"current setting recovers {100 * base.recovered.iloc[0]:.1f}% with "
              f"{base.cand_on_ok.iloc[0]:.2f} candidates/frame on tracked frames; "
              f"a grid point is only better if it lifts the first without lifting "
              f"the second, since every extra candidate is another way for pick() "
              f"to go wrong")

    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        P = G.pivot(index="min_depth", columns="dark_thr", values="recovered")
        Q = G.pivot(index="min_depth", columns="dark_thr", values="cand_on_ok")
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, M, t, cm in ((axs[0], P, "fraction of misses recovered", "viridis"),
                             (axs[1], Q, "candidates/frame on tracked frames", "magma")):
            im = ax.imshow(M.values, origin="lower", aspect="auto", cmap=cm)
            ax.set(xticks=range(len(M.columns)), xticklabels=M.columns,
                   yticks=range(len(M.index)), yticklabels=M.index,
                   xlabel="dark_thr", ylabel="min_depth", title=t)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(a.fig, dpi=150)
        print(f"wrote {a.fig}")


# ---------------------------------------------------------------- clips

def cmd_clips(a):
    T, F = load_failures(a.track, a.audit)
    if F.empty:
        print("no misses -- nothing to clip")
        return
    runs = (F.groupby("run_start")
             .agg(run_len=("run_len", "first"), cause=("cause", "first"),
                  prev_x=("prev_x", "first"), prev_y=("prev_y", "first"))
             .sort_values("run_len", ascending=False).head(a.k).reset_index())
    bg, mask, g = get_bg_mask(a)
    edge = cv2.dilate(mask, np.ones((3, 3), np.uint8)) - mask
    cap, N = bt.open_video(a.video)
    frame_col = T.frame.to_numpy(int) if "frame" in T.columns else np.arange(len(T))
    writer = None
    for _, r in runs.iterrows():
        s0 = int(frame_col[int(r.run_start)])
        lo, hi = max(0, s0 - a.pad), min(N, s0 + int(r.run_len) + a.pad)
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        for f in range(lo, hi):
            ok, im = cap.read()
            if not ok:
                break
            grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
            cands, _, _ = bt.candidates(grey, bg, mask, a.dark_thr,
                                        a.area[0], a.area[1], a.min_depth)
            vis = im.copy()
            vis[edge > 0] = (0, 160, 0)
            for c in cands:
                cv2.circle(vis, (int(c["x"]), int(c["y"])), 6, (255, 160, 0), 1)
                cv2.putText(vis, f"{c['score']:.0f}", (int(c["x"]) + 7, int(c["y"])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 160, 0), 1)
            inside = s0 <= f < s0 + int(r.run_len)
            if np.isfinite(r.prev_x):
                cv2.circle(vis, (int(r.prev_x), int(r.prev_y)), 18,
                           (0, 0, 255) if inside else (0, 255, 255), 1)
            cv2.putText(vis, f"{f} {'MISS ' + r.cause if inside else 'tracked'}"
                             f"  cands={len(cands)}", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255) if inside else (40, 40, 40), 1, cv2.LINE_AA)
            if writer is None:
                writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                         a.fps, (vis.shape[1], vis.shape[0]))
            writer.write(vis)
    cap.release()
    if writer is not None:
        writer.release()
        print(f"wrote {a.out}: {len(runs)} runs, pad {a.pad} frames either side. "
              f"Orange circles are every candidate the tracker saw, scored; the "
              f"red ring is the last accepted position.")


# ---------------------------------------------------------------- cli

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q, video=True):
        if video:
            q.add_argument("--video", required=True)
        q.add_argument("--track", required=True, help="blob_tracker track CSV")
        q.add_argument("--audit", default=None, help="optional audit run table")
        q.add_argument("--seed", type=int, default=0)

    def seg_args(q):
        q.add_argument("--geom", required=True)
        q.add_argument("--bg", default=None, help="cached <mask>_bg.npy -- use it")
        q.add_argument("--bg-stride", type=int, default=900)
        q.add_argument("--sat-thr", type=float, default=245)
        q.add_argument("--erode", type=int, default=0)
        q.add_argument("--dark-thr", type=float, default=200)
        q.add_argument("--area", type=int, nargs=2, default=[200, 3000])
        q.add_argument("--min-depth", type=float, default=40)

    f = sub.add_parser("frames", help="contact sheet of failure frames")
    common(f)
    f.add_argument("--n", type=int, default=48)
    f.add_argument("--cols", type=int, default=8)
    f.add_argument("--tile", type=int, default=240)
    f.add_argument("--out", default="miss_sheet.png")
    f.add_argument("--outdir", default=None, help="also write the frames singly")
    f.set_defaults(func=cmd_frames)

    r = sub.add_parser("probe", help="is a real blob there, just outside the gate?")
    common(r); seg_args(r)
    r.add_argument("--max-step", type=float, default=12.0)
    r.add_argument("--n", type=int, default=400)
    r.add_argument("--out", default="miss_probe.csv")
    r.add_argument("--fig", default=None)
    r.set_defaults(func=cmd_probe)

    s = sub.add_parser("sweep", help="threshold grid on miss frames and controls")
    common(s); seg_args(s)
    s.add_argument("--dark-grid", type=float, nargs="+",
                   default=[180, 190, 200, 210, 220])
    s.add_argument("--depth-grid", type=float, nargs="+", default=[20, 30, 40, 50])
    s.add_argument("--n", type=int, default=250)
    s.add_argument("--n-control", type=int, default=250)
    s.add_argument("--out", default="miss_sweep.csv")
    s.add_argument("--fig", default=None)
    s.set_defaults(func=cmd_sweep)

    c = sub.add_parser("clips", help="mp4 around the worst runs, candidates drawn")
    common(c); seg_args(c)
    c.add_argument("--k", type=int, default=6, help="longest runs to include")
    c.add_argument("--pad", type=int, default=45)
    c.add_argument("--fps", type=float, default=15.0)
    c.add_argument("--out", default="miss_clips.mp4")
    c.set_defaults(func=cmd_clips)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
