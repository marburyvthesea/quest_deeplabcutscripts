#!/usr/bin/env python
"""
Detect T-maze door open/closed state from video, per frame.

Principle (John Marshall's observation, 2026-09-19): the door panel is white and
saturated, so the door itself carries almost no contrast. But the door slides
vertically in rails cut into the maze wall; when the door lifts, the exposed
rails show bare metal and shadow and become DARK. A saturation problem thus
becomes a darkening problem.

Two facts make this work where a plain ROI mean fails:

  1. The rail pixels are a small fraction of any hand-drawn door ROI (~35 of
     456 px in the original FIJI ROI), so an ROI mean dilutes a 20-level rail
     signal down to ~1 level. The rail mask must be LEARNED, not drawn.

  2. The mouse runs through the open doorway, and the animal is far darker
     than the rails (drops the ROI to ~85 vs ~227 for an open door). Rejecting
     frames where the animal is present would delete the middle of every real
     opening. Instead we measure darkening ON the rails and IN THE SURROUND
     separately: the door darkens only the rails, the animal darkens both. When
     the surround is dark the animal is occluding, and the door state is HELD
     at its last confident value rather than forced closed.

Mask learning needs one verified open window and one or more closed windows.
Per-pixel MEDIAN over each window is essential: the mean lets brief animal
transits into the mask (they are ~20-40 frames long and can dominate a mean).

NOTE ON VIDEO CHOICE: run this on the RAW behaviour video (C.avi). A DLC
*labeled* mp4 has coloured keypoint markers drawn on it that turn dark in
greyscale and can land inside the door ROI. The script measures how much
colour it sees in the ROI and warns if the input looks like a labeled video.

Usage
-----
  python door_state.py --video C.avi --geom maze_geometry_C.json --door DS0 \
      --open-window 5175 5310 --closed-window 4880 5120 --closed-window 5420 5700 \
      --out C_doorstate.csv --qc C_door_qc.png

If --open-window is omitted, the first 'fall'/'raise_' pair in the geometry
file's door_log_frames is used to propose one (fall_end+14 .. raise_start-14).
"""
import argparse
import json
import sys

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import medfilt

FPS_DEFAULT = 30.0


def read_crop(cap, a, b, box, want_colour=False):
    """Greyscale crop for frames [a, b). Returns (stack, colourness)."""
    x0, x1, y0, y1 = box
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(a))
    out, col = [], 0.0
    for _ in range(int(b) - int(a)):
        ok, im = cap.read()
        if not ok:
            break
        sub = im[y0:y1, x0:x1]
        out.append(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY))
        if want_colour:
            s = sub.astype(np.int16)
            col = max(col, float((s.max(2) - s.min(2)).mean()))
    if not out:
        raise SystemExit(f"no frames read at {a}..{b}")
    return np.stack(out).astype(np.float32), col


def learn_mask(cap, box, door_rc, open_win, closed_wins, thr):
    """Per-pixel median difference closed-minus-open, restricted to the door box."""
    o = np.median(read_crop(cap, *open_win, box)[0], axis=0)
    cl = [np.median(read_crop(cap, *w, box)[0], axis=0) for w in closed_wins]
    ref = np.minimum.reduce(cl) if len(cl) > 1 else cl[0]
    delta = ref - o
    inbox = np.zeros(delta.shape, bool)
    inbox[door_rc[0]:door_rc[1], door_rc[2]:door_rc[3]] = True
    mask = (delta > thr) & inbox
    mask = ndi.binary_opening(mask, np.ones((2, 2)))
    return mask, inbox, delta, ref


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--geom", required=True)
    p.add_argument("--door", default="DS0")
    p.add_argument("--pad", type=int, default=12, help="context px around door box")
    p.add_argument("--open-window", type=int, nargs=2, metavar=("A", "B"))
    p.add_argument("--closed-window", type=int, nargs=2, action="append",
                   metavar=("A", "B"), default=[])
    p.add_argument("--mask-thr", type=float, default=12.0,
                   help="min median darkening (grey levels) for a rail pixel")
    p.add_argument("--open-thr", type=float, default=14.0)
    p.add_argument("--close-thr", type=float, default=8.0)
    p.add_argument("--occl-thr", type=float, default=6.0,
                   help="surround darkening above which the animal is occluding")
    p.add_argument("--smooth", type=int, default=9, help="median filter, odd frames")
    p.add_argument("--min-frames", type=int, default=8)
    p.add_argument("--fps", type=float, default=FPS_DEFAULT)
    p.add_argument("--out", required=True)
    p.add_argument("--qc")
    a = p.parse_args()

    g = json.load(open(a.geom))
    d = g["doors"][a.door]
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fsz = g.get("frame_size_px")
    if fsz and [W, H] != list(fsz):
        raise SystemExit(f"video is {W}x{H} but geometry says {fsz}; ROIs would be offset")

    x0, x1 = max(0, d["x"] - a.pad), min(W, d["x"] + d["w"] + a.pad)
    y0, y1 = max(0, d["y"] - a.pad), min(H, d["y"] + d["h"] + a.pad)
    box = (x0, x1, y0, y1)
    door_rc = (d["y"] - y0, d["y"] + d["h"] - y0, d["x"] - x0, d["x"] + d["w"] - x0)

    ow = a.open_window
    cws = a.closed_window
    if ow is None:
        log = g.get("door_log_frames", {})
        if not (log.get("fall") and log.get("raise_")):
            raise SystemExit("no --open-window and no door_log_frames in geometry")
        f, r = log["fall"][0], log["raise_"][0]
        ow = [f[1] + 14, r[0] - 14]
        print(f"open window from door_log_frames event 1: {ow}", file=sys.stderr)
    if not cws:
        log = g.get("door_log_frames", {})
        f, r = log["fall"][0], log["raise_"][0]
        cws = [[max(0, f[0] - 260), f[0] - 14], [r[1] + 14, r[1] + 300]]
        print(f"closed windows inferred: {cws}", file=sys.stderr)

    mask, inbox, delta, ref = learn_mask(cap, box, door_rc, ow, cws, a.mask_thr)
    if mask.sum() < 8:
        raise SystemExit(f"only {mask.sum()} rail pixels found; check --open-window "
                         f"and --mask-thr (door may not be open in that window)")
    ys, xs = np.nonzero(mask)
    print(f"rail mask: {mask.sum()} px, x {x0+xs.min()}..{x0+xs.max()} "
          f"y {y0+ys.min()}..{y0+ys.max()}, mean darkening {delta[mask].mean():.1f}")
    off = inbox & ~mask

    # full pass
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    on_t, off_t, colour = [], [], 0.0
    k = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        sub = im[y0:y1, x0:x1]
        gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dif = ref - gray
        on_t.append(dif[mask].mean())
        off_t.append(dif[off].mean())
        if k % 5000 == 0:
            s = sub.astype(np.int16)
            colour = max(colour, float((s.max(2) - s.min(2)).mean()))
            print(f"  {k}/{N}", file=sys.stderr, flush=True)
        k += 1
    on_t, off_t = np.array(on_t), np.array(off_t)
    if colour > 2.0:
        print(f"WARNING: door ROI is not greyscale (colourness {colour:.1f}). This looks "
              f"like a DLC labeled video; overlay markers may cause false openings. "
              f"Re-run on the raw behaviour video.", file=sys.stderr)

    sm = max(1, a.smooth | 1)
    on_s, off_s = medfilt(on_t, sm), medfilt(off_t, sm)
    occl = off_s > a.occl_thr
    state = np.zeros(len(on_s), np.int8)
    cur = 0
    for j in range(len(on_s)):
        if not occl[j]:
            cur = 1 if on_s[j] > a.open_thr else (0 if on_s[j] < a.close_thr else cur)
        state[j] = cur

    lab, n = ndi.label(state == 1)
    for j in range(1, n + 1):
        w = np.nonzero(lab == j)[0]
        if len(w) < a.min_frames:
            state[w] = 0
    lab, n = ndi.label(state == 1)
    ivs = [(int(np.nonzero(lab == j)[0].min()), int(np.nonzero(lab == j)[0].max()))
           for j in range(1, n + 1)]

    import csv
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "on_rail", "surround", "occluded", "door_open"])
        for j in range(len(state)):
            w.writerow([j, round(float(on_t[j]), 2), round(float(off_t[j]), 2),
                        int(occl[j]), int(state[j])])
    print(f"wrote {a.out}: {len(state)} frames, {n} open intervals, "
          f"{state.mean()*100:.2f}% of frames open, "
          f"{occl.mean()*100:.1f}% occluded by animal")
    for s, e in ivs[:40]:
        print(f"  open {s}..{e}  {e-s+1} fr  {(e-s+1)/a.fps:.1f} s")
    if n > 40:
        print(f"  ... {n-40} more")

    if a.qc:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(13, 8))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(ref, cmap="gray", vmin=0, vmax=255, extent=[x0, x1, y1, y0])
        ax.set_title("closed reference")
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(delta, cmap="inferno", vmin=0, vmax=30, extent=[x0, x1, y1, y0])
        ax2.contour(mask, levels=[.5], colors="cyan", linewidths=.8,
                    extent=[x0, x1, y1, y0], origin="upper")
        ax2.set_title(f"closed − open; cyan = rail mask ({mask.sum()} px)")
        for q in (ax, ax2):
            q.add_patch(plt.Rectangle((d["x"], d["y"]), d["w"], d["h"],
                                      ec="w", fc="none", lw=1))
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(on_t, lw=.3, color="tab:red", label="on-rail")
        ax3.plot(off_t, lw=.3, color="tab:blue", label="surround")
        ax3.axhline(a.open_thr, color="tab:red", ls=":", lw=.8)
        ax3.axhline(a.occl_thr, color="tab:blue", ls=":", lw=.8)
        ax3.set_ylim(-6, 60); ax3.legend(fontsize=8)
        ax3.set_ylabel("Δ intensity vs closed")
        ax4 = fig.add_subplot(gs[2, :], sharex=ax3)
        ax4.fill_between(np.arange(len(state)), 0, state, step="mid",
                         color="tab:orange", lw=0)
        ax4.set_yticks([0, 1], ["closed", "open"]); ax4.set_xlabel("frame")
        ax4.set_title(f"{n} open intervals")
        fig.tight_layout(); fig.savefig(a.qc, dpi=130)
        print(f"wrote {a.qc}")


if __name__ == "__main__":
    main()
