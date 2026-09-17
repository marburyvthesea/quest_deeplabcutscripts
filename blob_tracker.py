#!/usr/bin/env python
"""
Track a dark animal on a saturated floor, with the search mask taken from the
video's own background rather than from hand-drawn ROIs.

Why a derived mask: in these T-maze recordings the floor is blown out (median
253/255) while the maze frame contains open voids with cable tangles beneath.
Hand-drawn ROIs cannot separate those voids from the running surface, and the
tangle is the most animal-shaped dark object in the image -- it is what defeats
SuperAnimal here. Thresholding the background at near-saturation isolates the
running surface exactly, and on that surface the animal is the only thing that
darkens it.

Candidates are scored on how much darker than background they are, how well
they fill their bounding box, and how elongated they are, so that cable
segments crossing the floor lose to the animal. A constant-velocity gate then
rejects candidates that would require an implausible jump.

  # build the mask and check it before tracking anything
  python blob_tracker.py mask --video C.avi --geom maze_geometry_C.json \
         --out C_mask.png

  # tune on a window, with an overlay to watch
  python blob_tracker.py track --video C.avi --geom maze_geometry_C.json \
         --start 5100 --n 500 --overlay C_qc_5100.mp4 --out C_win5100.csv

  # full session
  python blob_tracker.py track --video C.avi --geom maze_geometry_C.json \
         --out C_blobtrack.csv --fig C_blobtrack.png
"""
import argparse
import json
import sys

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- background

def background(cap, n_frames, stride, verbose=True):
    """Median of frames sampled every `stride`, as float32 grey."""
    samp = []
    for f in range(0, n_frames, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if ok:
            samp.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    if not samp:
        raise SystemExit("could not read any frames for the background model")
    if verbose:
        print(f"background from {len(samp)} sampled frames", flush=True)
    return np.median(np.stack(samp), 0).astype(np.float32)


def surface_mask(bg, geom, sat_thr=245, erode=0, close=5):
    """Runnable surface = near-saturated background inside the maze rectangle."""
    H, W = bg.shape
    m = (bg >= sat_thr).astype(np.uint8)
    b = geom["maze_outer"]
    keep = np.zeros_like(m)
    keep[max(0, b["y"]):b["y"] + b["h"], max(0, b["x"]):b["x"] + b["w"]] = 1
    m *= keep
    # the voids are mostly dark and drop out on their own, but equipment visible
    # through them can be saturated, so the hand-measured islands are subtracted
    # explicitly -- the one piece of information the video cannot supply
    for b in geom.get("exclude_islands", []):
        m[max(0, b["y"]):b["y"] + b["h"], max(0, b["x"]):b["x"] + b["w"]] = 0
    if close > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    if erode > 0:
        m = cv2.erode(m, np.ones((erode, erode), np.uint8))
    for b in geom.get("exclude_islands", []):  # again, so closing cannot bridge in
        m[max(0, b["y"]):b["y"] + b["h"], max(0, b["x"]):b["x"] + b["w"]] = 0
    return m


# ---------------------------------------------------------------- candidates

def candidates(grey, bg, mask, dark_thr, amin, amax, min_depth=40, open_k=3, close_k=7):
    """Dark connected components on the surface, with shape/darkness features."""
    d = ((grey < dark_thr).astype(np.uint8)) * mask
    if open_k:
        d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    if close_k:
        d = cv2.morphologyEx(d, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    nl, lab, st, cen = cv2.connectedComponentsWithStats(d, 8)
    out = []
    for i in range(1, nl):
        a = int(st[i, cv2.CC_STAT_AREA])
        if not (amin <= a <= amax):
            continue
        x, y = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
        w, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        sel = lab[y:y + h, x:x + w] == i
        depth = float((bg[y:y + h, x:x + w][sel] - grey[y:y + h, x:x + w][sel]).mean())
        if depth < min_depth:
            # dark in the background model too, so it is furniture, not the animal
            continue
        extent = a / float(w * h)
        ys, xs = np.nonzero(sel)
        if ys.size >= 5:
            c = np.cov(np.vstack([xs, ys]))
            ev = np.sort(np.linalg.eigvalsh(c))[::-1]
            elong = float(np.sqrt(max(ev[0], 1e-6) / max(ev[1], 1e-6)))
        else:
            elong = 1.0
        # dark and compact wins; long thin things (cables) are penalised
        score = depth * extent / (1.0 + max(0.0, elong - 3.0))
        out.append(dict(x=float(cen[i][0]), y=float(cen[i][1]), area=a, depth=depth,
                        extent=extent, elong=elong, score=score, label=i))
    return out, d, lab


def pick(cands, prev, max_step, gap, jump_penalty=0.5):
    """Best candidate given the previous accepted position (constant-velocity gate)."""
    if not cands:
        return None
    if prev is None:
        return max(cands, key=lambda c: c["score"])
    allow = max_step * max(1, gap + 1)
    viable = []
    for c in cands:
        dist = float(np.hypot(c["x"] - prev[0], c["y"] - prev[1]))
        if dist <= allow:
            viable.append((c["score"] * (1.0 - jump_penalty * dist / max(allow, 1e-6)), dist, c))
    if not viable:
        return None
    s, dist, c = max(viable, key=lambda t: t[0])
    c = dict(c)
    c["dist"] = dist
    return c


# ---------------------------------------------------------------- commands

def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    return cap, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def cmd_mask(a):
    g = json.load(open(a.geom))
    cap, N = open_video(a.video)
    bg = background(cap, N, a.bg_stride)
    m = surface_mask(bg, g, a.sat_thr, a.erode)
    print(f"frame {bg.shape[1]}x{bg.shape[0]}, {N} frames")
    print(f"surface mask: {int(m.sum())} px "
          f"({100*m.sum()/(bg.shape[0]*bg.shape[1]):.1f}% of frame)")
    print(f"background on surface: median {np.median(bg[m>0]):.0f}, "
          f"{100*(bg[m>0]>=250).mean():.0f}% at >=250")
    vis = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    vis[m > 0] = (0.55 * vis[m > 0] + np.array([0, 90, 0])).astype(np.uint8)
    for b in g.get("exclude_islands", []) + [g["maze_outer"]]:
        cv2.rectangle(vis, (b["x"], b["y"]), (b["x"]+b["w"], b["y"]+b["h"]), (0, 0, 255), 1)
    cv2.imwrite(a.out, vis)
    np.save(a.out.replace(".png", "_bg.npy"), bg)
    print(f"wrote {a.out} and {a.out.replace('.png', '_bg.npy')}")


def cmd_track(a):
    g = json.load(open(a.geom))
    cap, N = open_video(a.video)
    n = N if a.n is None else min(a.n, N - a.start)
    bg = (np.load(a.bg) if a.bg else background(cap, N, a.bg_stride))
    m = surface_mask(bg, g, a.sat_thr, a.erode)
    print(f"tracking frames {a.start}-{a.start+n} of {N}; surface {int(m.sum())} px", flush=True)

    writer = None
    if a.overlay:
        writer = cv2.VideoWriter(a.overlay, cv2.VideoWriter_fourcc(*"mp4v"),
                                 a.fps, (bg.shape[1], bg.shape[0]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
    rows, prev, gap, trail = [], None, 0, []
    for k in range(n):
        ok, im = cap.read()
        if not ok:
            break
        grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        cands, dmask, lab = candidates(grey, bg, m, a.dark_thr, a.area[0], a.area[1],
                                       a.min_depth)
        c = pick(cands, prev, a.max_step, gap, a.jump_penalty)
        if c is None:
            rows.append((a.start + k, np.nan, np.nan, 0, np.nan, np.nan, np.nan, np.nan,
                         len(cands), "miss"))
            gap += 1
        else:
            rows.append((a.start + k, c["x"], c["y"], c["area"], c["depth"], c["extent"],
                         c["elong"], c["score"], len(cands), "ok"))
            prev, gap = (c["x"], c["y"]), 0
            trail.append((int(c["x"]), int(c["y"])))
        if writer is not None:
            vis = im.copy()
            vis[(dmask > 0)] = (0.4 * vis[(dmask > 0)] + np.array([0, 0, 120])).astype(np.uint8)
            if c is not None:
                cv2.circle(vis, (int(c["x"]), int(c["y"])), 6, (0, 255, 255), 2)
                if c["label"]:
                    cnt, _ = cv2.findContours((lab == c["label"]).astype(np.uint8),
                                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, cnt, -1, (0, 255, 0), 1)
            for p, q in zip(trail[-120:-1], trail[-119:]):
                cv2.line(vis, p, q, (255, 200, 0), 1)
            cv2.putText(vis, f"{a.start+k}  cands={len(cands)}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            writer.write(vis)
        if a.report_every and (k + 1) % a.report_every == 0:
            print(f"  {k+1}/{n}", flush=True)
    if writer is not None:
        writer.release()
        print(f"wrote {a.overlay}")

    T = pd.DataFrame(rows, columns=["frame", "x", "y", "area", "depth", "extent",
                                    "elong", "score", "n_cand", "status"])
    # fill short gaps, flag them, then derive speed from lightly smoothed position
    xs, ys = T.x.to_numpy(copy=True), T.y.to_numpy(copy=True)
    isnan = ~np.isfinite(xs)
    T["interp"] = False
    if a.max_gap > 0 and isnan.any():
        e = np.flatnonzero(np.diff(np.concatenate(([0], isnan.view(np.int8), [0]))))
        for s0, s1 in zip(e[::2], e[1::2]):
            if s1 - s0 <= a.max_gap and s0 > 0 and s1 < len(T):
                for arr in (xs, ys):
                    arr[s0:s1] = np.interp(np.arange(s0, s1), [s0 - 1, s1],
                                           [arr[s0 - 1], arr[s1]])
                T.loc[s0:s1 - 1, "interp"] = True
        T["x"], T["y"] = xs, ys
    sm = T[["x", "y"]].rolling(a.smooth, center=True, min_periods=1).median()
    T["speed_px"] = np.hypot(sm.x.diff(), sm.y.diff())
    T.to_csv(a.out, index=False)

    det = np.isfinite(T.x).mean()
    print(f"\ndetected {100*det:.1f}% of frames "
          f"({int(T.interp.sum())} filled by interpolation)")
    print(f"candidates per frame: median {T.n_cand.median():.0f}, max {T.n_cand.max()}")
    print(f"area median {T.area[T.area>0].median():.0f} px; "
          f"depth median {T.depth.median():.0f} grey levels; "
          f"extent median {T.extent.median():.2f}")
    sp = T.speed_px.to_numpy()
    sp = sp[np.isfinite(sp)]
    if sp.size:
        print(f"speed px/frame: median {np.median(sp):.2f}, 95th {np.percentile(sp,95):.1f}, "
              f"max {sp.max():.1f}; path {np.nansum(sp):.0f} px")
    print(f"wrote {a.out}")

    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        ok = np.isfinite(T.x)
        hb = ax[0].hexbin(T.x[ok], T.y[ok], gridsize=60, bins="log", cmap="magma",
                          extent=(g["maze_outer"]["x"], g["maze_outer"]["x"]+g["maze_outer"]["w"],
                                  g["maze_outer"]["y"], g["maze_outer"]["y"]+g["maze_outer"]["h"]))
        for b in g.get("exclude_islands", []):
            ax[0].add_patch(plt.Rectangle((b["x"], b["y"]), b["w"], b["h"], fill=False,
                                          ec="#4dd0e1", lw=1.2))
        ax[0].invert_yaxis(); ax[0].set_aspect("equal")
        ax[0].set_xlabel("x (px)"); ax[0].set_ylabel("y (px)")
        ax[0].set_title("occupancy", loc="left")
        fig.colorbar(hb, ax=ax[0], label="frames")
        rm = T.speed_px.rolling(int(a.fps), center=True, min_periods=5).median()
        ax[1].plot(T.frame / a.fps / 60.0, rm, lw=0.5, color="#4c72b0")
        ax[1].set_xlabel("time (min)"); ax[1].set_ylabel("speed (px/frame, 1 s median)")
        ax[1].set_title("locomotion over the session", loc="left")
        fig.tight_layout(); fig.savefig(a.fig, dpi=200)
        print(f"wrote {a.fig}")


def cmd_post(a):
    """Flag stationary frames and identify recurring dwell sites (furniture)."""
    T = pd.read_csv(a.track)
    ok = np.isfinite(T.x)
    sd = (T[["x", "y"]].rolling(a.win, center=True, min_periods=a.win // 2).std()
          .max(axis=1).to_numpy())
    T["pos_sd"] = sd
    T["stationary"] = ok & (sd < a.sd_thr)
    # contiguous stationary episodes
    s = T.stationary.to_numpy().astype(np.int8)
    e = np.flatnonzero(np.diff(np.concatenate(([0], s, [0]))))
    eps = [(i, j) for i, j in zip(e[::2], e[1::2]) if j - i >= a.min_dwell]
    sites = []
    for i, j in eps:
        sites.append((float(T.x[i:j].median()), float(T.y[i:j].median()), i, j))
    # merge episodes whose medians sit within `link` px of each other
    merged = []
    for x, y, i, j in sites:
        for m in merged:
            if np.hypot(m["x"] - x, m["y"] - y) <= a.link:
                m["frames"] += j - i
                m["episodes"] += 1
                m["x"] = (m["x"] * (m["episodes"] - 1) + x) / m["episodes"]
                m["y"] = (m["y"] * (m["episodes"] - 1) + y) / m["episodes"]
                break
        else:
            merged.append(dict(x=x, y=y, frames=j - i, episodes=1))
    merged.sort(key=lambda m: -m["frames"])
    # A resting animal walked in: there is a continuous detected path from well
    # outside the site into it. A furniture lock is entered after losing the
    # animal, so the track teleports in across a run of missed frames.
    xa, ya = T.x.to_numpy(), T.y.to_numpy()
    miss = (T.status.to_numpy() == "miss")
    for m in merged:
        m["walked_in"] = 0
    for x, y, i, j in sites:
        site = min(merged, key=lambda m: np.hypot(m["x"] - x, m["y"] - y))
        far = np.flatnonzero((np.hypot(xa[:i] - x, ya[:i] - y) > a.approach_px) & ~miss[:i])
        if far.size:
            k = far[-1]
            if miss[k:i].sum() <= a.max_approach_gap:
                site["walked_in"] += 1
    T["furniture"] = False
    print(f"stationary frames: {int(T.stationary.sum())} of {int(ok.sum())} detected "
          f"({100*T.stationary.sum()/max(ok.sum(),1):.1f}%)")
    print(f"dwell episodes >= {a.min_dwell} frames: {len(eps)}; distinct sites: {len(merged)}")
    print("\n     x      y   episodes  walked-in   frames   seconds   verdict")
    for m in merged[:12]:
        furn = m["walked_in"] < a.min_walked_frac * m["episodes"]
        if furn:
            d = np.hypot(T.x - m["x"], T.y - m["y"])
            T.loc[T.stationary & (d <= a.link), "furniture"] = True
        print(f"  {m['x']:5.0f}  {m['y']:5.0f}   {m['episodes']:8d} {m['walked_in']:10d} "
              f"{m['frames']:8d} {m['frames']/a.fps:9.1f}   "
              f"{'furniture' if furn else 'animal at rest'}")
    print(f"\nframes marked furniture: {int(T.furniture.sum())} "
          f"({100*T.furniture.mean():.1f}% of session)")
    T["usable"] = ok & ~T.furniture
    print(f"usable detections: {int(T.usable.sum())} ({100*T.usable.mean():.1f}% of session)")
    T.to_csv(a.out, index=False)
    print(f"wrote {a.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--video", required=True)
        q.add_argument("--geom", required=True)
        q.add_argument("--sat-thr", type=float, default=245,
                       help="background >= this is the running surface")
        q.add_argument("--erode", type=int, default=0,
                       help="shrink the surface mask by this many px")
        q.add_argument("--bg-stride", type=int, default=900)

    m = sub.add_parser("mask", help="build and inspect the surface mask")
    common(m); m.add_argument("--out", default="mask.png"); m.set_defaults(func=cmd_mask)

    t = sub.add_parser("track", help="track the animal")
    common(t)
    t.add_argument("--bg", default=None, help="reuse a saved *_bg.npy")
    t.add_argument("--start", type=int, default=0)
    t.add_argument("--n", type=int, default=None)
    t.add_argument("--dark-thr", type=float, default=200)
    t.add_argument("--area", type=int, nargs=2, default=[200, 3000])
    t.add_argument("--min-depth", type=float, default=40,
                   help="reject blobs less than this many grey levels darker than background")
    t.add_argument("--max-step", type=float, default=12.0, help="px per frame gate")
    t.add_argument("--jump-penalty", type=float, default=0.5)
    t.add_argument("--max-gap", type=int, default=15, help="frames of gap to interpolate")
    t.add_argument("--smooth", type=int, default=5)
    t.add_argument("--fps", type=float, default=30.0)
    t.add_argument("--overlay", default=None)
    t.add_argument("--out", required=True)
    t.add_argument("--fig", default=None)
    t.add_argument("--report-every", type=int, default=20000)
    t.set_defaults(func=cmd_track)

    q = sub.add_parser("post", help="flag stationary frames and recurring dwell sites")
    q.add_argument("--track", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--win", type=int, default=151, help="frames for the rolling sd")
    q.add_argument("--sd-thr", type=float, default=4.0, help="px sd below which a frame is stationary")
    q.add_argument("--min-dwell", type=int, default=150, help="frames for an episode to count")
    q.add_argument("--link", type=float, default=15.0, help="px within which episodes are one site")
    q.add_argument("--approach-px", type=float, default=40.0,
                   help="distance the track must come from for an approach to count")
    q.add_argument("--max-approach-gap", type=int, default=5,
                   help="missed frames allowed during the approach")
    q.add_argument("--min-walked-frac", type=float, default=0.5,
                   help="fraction of a site's episodes that must be walked into")
    q.add_argument("--fps", type=float, default=30.0)
    q.set_defaults(func=cmd_post)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
