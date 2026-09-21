#!/usr/bin/env python
"""
Detect T-maze door open/closed state per frame, from the behaviour video alone.

WHY THIS WORKS
--------------
The door panel is white and saturates the camera, so the door itself carries
almost no contrast and no intensity measure of the door finds it. But the door
slides vertically in rails cut into the maze wall; when the door lifts, the
exposed rails show bare metal and shadow and go DARK. A saturation problem
becomes a darkening problem (J. Marshall, 2026-09-19).

THREE THINGS THAT DEFEAT THE OBVIOUS IMPLEMENTATION
---------------------------------------------------
1. ROI DILUTION. The exposed rail is a small fraction of any hand-drawn door
   ROI -- in session C, ~115 rail pixels inside a 456 px FIJI ROI. A 26 grey
   level rail signal averages down to +0.6 over the ROI, i.e. nothing. The rail
   mask must be LEARNED from an open/closed comparison, never drawn by hand.

2. THE ANIMAL. The mouse runs THROUGH the open doorway and is far darker than
   the rails (ROI drops to ~85 vs ~227 for an open door). Excluding frames
   where the animal is present would delete the middle of every real opening.
   Instead measure darkening ON the rails and in the SURROUND separately: the
   door darkens only the rails, the animal darkens both. When the surround
   darkens, the animal is occluding, and door state is HELD at its last
   confident value rather than forced closed. Without this hold, every opening
   is split into two intervals by the animal's own transit.

3. MEAN vs MEDIAN when learning. Animal transits are 20-40 frames long and can
   dominate a windowed mean, putting the animal's silhouette into the mask.
   Per-pixel median over each learning window rejects them.

CROSS-SESSION USE
-----------------
Lighting varies between sessions, so the mask and the reference are separated:

  * the RAIL MASK is geometry (where the rails are) and transfers between
    sessions of the same rig -- learn it once from example open/closed images
    and reuse with --mask;
  * the CLOSED REFERENCE is photometry (how bright the wall is today) and is
    ALWAYS recomputed from the target video itself, as a per-pixel median over
    frames sampled across the whole session. The door is closed the large
    majority of the time, so the median is the closed state by construction.

Detection thresholds default to fractions of the measured rail amplitude in
the target video, so they too follow the session's own contrast.

Slow illumination drift over a session is removed by subtracting a block-median
baseline (default 1200 frames) from the rail signal; that is far longer than
any opening, so real events survive it.

VIDEO CHOICE
------------
Run on the RAW behaviour video. A DeepLabCut *labeled* mp4 has coloured keypoint
markers drawn on it which turn dark in greyscale and can fall inside the door
ROI. The script measures colour in the ROI and warns if the input looks labeled.

USAGE
-----
First session of a rig -- learn the mask from a verified open/closed pair:

  python door_state.py --video C.avi --geom maze_geometry_C.json --door DS0 \
      --open-window 5175 5310 --closed-window 4880 5120 \
      --save-mask DS0_rail_mask.json \
      --out C_doorstate.csv --qc C_door_qc.png

Later sessions -- reuse the mask, reference recomputed automatically:

  python door_state.py --video D.avi --geom maze_geometry_D.json --door DS0 \
      --mask DS0_rail_mask.json --out D_doorstate.csv --qc D_door_qc.png

Learning from exported still images instead of frame windows (any session):

  python door_state.py --video D.avi --geom maze_geometry_D.json --door DS0 \
      --open-image open1.png --open-image open2.png \
      --closed-image closed1.png --closed-image closed2.png \
      --save-mask DS0_rail_mask.json --out D_doorstate.csv

OUTPUTS
-------
  <out>.csv              frame, on_rail, surround, occluded, door_open
  <out stem>_intervals.csv  interval index, start/end frame, duration, amplitude
  --qc PNG               reference, learned mask, signals, state raster
  --save-mask JSON       rail mask (RLE) + door box + provenance, for reuse
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import medfilt

FPS_DEFAULT = 30.0


# ----------------------------------------------------------------- utilities
def crop_box(geom, door, pad, W, H):
    d = geom["doors"][door]
    x0, x1 = max(0, d["x"] - pad), min(W, d["x"] + d["w"] + pad)
    y0, y1 = max(0, d["y"] - pad), min(H, d["y"] + d["h"] + pad)
    inbox_rc = (d["y"] - y0, d["y"] + d["h"] - y0, d["x"] - x0, d["x"] + d["w"] - x0)
    return d, (x0, x1, y0, y1), inbox_rc


def read_window(cap, a, b, box, cap_n=400):
    """Greyscale crops for frames [a, b), decimated to at most cap_n frames."""
    x0, x1, y0, y1 = box
    a, b = int(a), int(b)
    step = max(1, (b - a) // cap_n)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    out = []
    for i in range(b - a):
        ok, im = cap.read()
        if not ok:
            break
        if i % step == 0:
            out.append(cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY))
    if not out:
        raise SystemExit(f"no frames read at {a}..{b}")
    return np.stack(out).astype(np.float32)


def read_images(paths, box, W, H):
    x0, x1, y0, y1 = box
    out = []
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            raise SystemExit(f"cannot read image {p}")
        if (im.shape[1], im.shape[0]) != (W, H):
            raise SystemExit(f"{p} is {im.shape[1]}x{im.shape[0]} but video is {W}x{H}; "
                             f"ROI coordinates would be offset")
        out.append(cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY))
    return np.stack(out).astype(np.float32)


def sampled_reference(cap, box, n_frames, n_sample=1200):
    """Per-pixel median over frames spread across the session = closed state."""
    x0, x1, y0, y1 = box
    idx = np.unique(np.linspace(0, n_frames - 1, min(n_sample, n_frames)).astype(int))
    out = []
    for f in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, im = cap.read()
        if ok:
            out.append(cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY))
    if not out:
        raise SystemExit("could not sample frames for the closed reference")
    return np.median(np.stack(out).astype(np.float32), axis=0), len(out)


def rle_encode(mask):
    flat = mask.ravel().astype(np.uint8)
    d = np.diff(np.concatenate(([0], flat, [0])))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return [[int(s), int(e - s)] for s, e in zip(starts, ends)]


def rle_decode(runs, shape):
    flat = np.zeros(int(np.prod(shape)), np.uint8)
    for s, ln in runs:
        flat[s:s + ln] = 1
    return flat.reshape(shape).astype(bool)


def block_baseline(x, block):
    """Median in non-overlapping blocks, linearly interpolated. Removes slow drift."""
    n = len(x)
    if block <= 1 or n < 2 * block:
        return np.full(n, float(np.median(x)))
    edges = np.arange(0, n + 1, block)
    if edges[-1] != n:
        edges = np.append(edges, n)
    cen = (edges[:-1] + edges[1:]) / 2.0
    med = np.array([np.median(x[a:b]) for a, b in zip(edges[:-1], edges[1:])])
    return np.interp(np.arange(n), cen, med)


# ------------------------------------------------------------ mask learning
def learn_mask(open_stack, closed_stack, inbox_rc, thr):
    o = np.median(open_stack, axis=0)
    c = np.median(closed_stack, axis=0)
    delta = c - o                                    # positive = darker when open
    inbox = np.zeros(delta.shape, bool)
    r0, r1, c0, c1 = inbox_rc
    inbox[r0:r1, c0:c1] = True
    mask = (delta > thr) & inbox
    mask = ndi.binary_opening(mask, np.ones((2, 2)))
    return mask, inbox, delta


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="RAW behaviour video")
    p.add_argument("--geom", required=True, help="maze geometry json with doors{}")
    p.add_argument("--door", default="DS0")
    p.add_argument("--pad", type=int, default=12, help="context px around the door box")

    src = p.add_argument_group("mask source (one of)")
    src.add_argument("--mask", help="reuse a mask saved by --save-mask")
    src.add_argument("--open-window", type=int, nargs=2, metavar=("A", "B"))
    src.add_argument("--closed-window", type=int, nargs=2, action="append",
                     metavar=("A", "B"), default=[])
    src.add_argument("--open-image", action="append", default=[],
                     help="still with the door OPEN; repeatable; any session, same rig")
    src.add_argument("--closed-image", action="append", default=[],
                     help="still with the door CLOSED; repeatable")
    src.add_argument("--save-mask", help="write the learned mask here for reuse")

    tun = p.add_argument_group("tuning (defaults scale to measured rail amplitude)")
    tun.add_argument("--mask-thr", type=float, default=12.0,
                     help="min median darkening for a rail pixel (grey levels)")
    tun.add_argument("--open-frac", type=float, default=0.45,
                     help="open threshold as fraction of rail amplitude")
    tun.add_argument("--close-frac", type=float, default=0.28)
    tun.add_argument("--occl-frac", type=float, default=0.22,
                     help="surround darkening (fraction of amplitude) = animal present")
    tun.add_argument("--smooth", type=int, default=9, help="median filter, odd frames")
    tun.add_argument("--drift-block", type=int, default=1200,
                     help="block length for drift baseline; 0 disables")
    tun.add_argument("--min-frames", type=int, default=8)
    tun.add_argument("--fps", type=float, default=FPS_DEFAULT)
    tun.add_argument("--ref-sample", type=int, default=1200)

    p.add_argument("--out", required=True)
    p.add_argument("--qc")
    a = p.parse_args()

    geom = json.load(open(a.geom))
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fsz = geom.get("frame_size_px")
    if fsz and [W, H] != list(fsz):
        raise SystemExit(f"video is {W}x{H} but geometry says {fsz}; ROIs would be offset")
    d, box, inbox_rc = crop_box(geom, a.door, a.pad, W, H)
    x0, x1, y0, y1 = box

    # ---- closed reference: always from the target video (handles lighting) ----
    ref, n_ref = sampled_reference(cap, box, N, a.ref_sample)
    print(f"closed reference from {n_ref} sampled frames of this video")

    # ---- rail mask ----
    if a.mask:
        mj = json.load(open(a.mask))
        if mj["box"] != [x0, x1, y0, y1]:
            raise SystemExit(f"saved mask box {mj['box']} != current {[x0, x1, y0, y1]}; "
                             f"same --door and --pad are required to reuse a mask")
        mask = rle_decode(mj["rle"], tuple(mj["shape"]))
        delta = None
        print(f"reusing mask from {a.mask}: {mask.sum()} px "
              f"(learned on {mj.get('learned_from', 'unknown')})")
        inbox = np.zeros(mask.shape, bool)
        r0, r1, c0, c1 = inbox_rc
        inbox[r0:r1, c0:c1] = True
    else:
        if a.open_image:
            op = read_images(a.open_image, box, W, H)
            cl = (read_images(a.closed_image, box, W, H)
                  if a.closed_image else ref[None])
            prov = f"{len(a.open_image)} open / " \
                   f"{len(a.closed_image) if a.closed_image else 'ref'} closed images"
        else:
            ow = a.open_window
            if ow is None:
                log = geom.get("door_log_frames", {})
                if not (log.get("fall") and log.get("raise_")):
                    raise SystemExit(
                        "need one of --mask, --open-window, or --open-image "
                        "(no door_log_frames in geometry to fall back on)")
                f, r = log["fall"][0], log["raise_"][0]
                ow = [f[1] + 14, r[0] - 14]
                print(f"open window taken from door_log_frames event 1: {ow}")
            op = read_window(cap, ow[0], ow[1], box)
            cl = (np.concatenate([read_window(cap, w[0], w[1], box)
                                  for w in a.closed_window])
                  if a.closed_window else ref[None])
            prov = f"open frames {ow}, closed {a.closed_window or 'session reference'}"
        mask, inbox, delta = learn_mask(op, cl, inbox_rc, a.mask_thr)
        if mask.sum() < 8:
            raise SystemExit(
                f"only {mask.sum()} rail pixels found. The door is probably not open in "
                f"the supplied open example, or --mask-thr ({a.mask_thr}) is too high.")
        ys, xs = np.nonzero(mask)
        print(f"rail mask: {mask.sum()} px, x {x0+xs.min()}..{x0+xs.max()} "
              f"y {y0+ys.min()}..{y0+ys.max()}, mean darkening {delta[mask].mean():.1f}")
        if a.save_mask:
            json.dump({"door": a.door, "box": [x0, x1, y0, y1],
                       "shape": list(mask.shape), "rle": rle_encode(mask),
                       "learned_from": prov, "video": os.path.basename(a.video),
                       "pad": a.pad}, open(a.save_mask, "w"), indent=1)
            print(f"wrote {a.save_mask}")

    off = inbox & ~mask
    if off.sum() < 8:
        raise SystemExit("surround region is too small; increase --pad")

    # ---- full pass ----
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    on_t, off_t, colour, k = [], [], 0.0, 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        sub = im[y0:y1, x0:x1]
        dif = ref - cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).astype(np.float32)
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
              f"like a DeepLabCut labeled video; overlay markers may cause false "
              f"openings. Re-run on the raw behaviour video.", file=sys.stderr)

    # ---- drift removal and thresholds ----
    base = block_baseline(on_t, a.drift_block) if a.drift_block else 0.0
    on_c = on_t - base
    sm = max(1, a.smooth | 1)
    on_s, off_s = medfilt(on_c, sm), medfilt(off_t, sm)
    # Rail amplitude is measured from the TARGET video in both learn and reuse
    # mode, so the two give identical results and thresholds track this
    # session's lighting. Frames where the surround is elevated are animal
    # transits; excluding them leaves door openings as the top of the
    # on-rail distribution.
    quiet = off_s < np.percentile(off_s, 90)
    amp = float(np.percentile(on_s[quiet], 99.0))
    if amp < 4.0:
        print(f"WARNING: measured rail amplitude is only {amp:.1f} grey levels. The "
              f"door may never open in this video, or the mask may be misplaced.",
              file=sys.stderr)
    if delta is not None:
        learned = float(delta[mask].mean())
        print(f"rail amplitude: {amp:.1f} measured in this video "
              f"({learned:.1f} in the learning windows)")
        if learned > 0 and not 0.5 < amp / learned < 2.0:
            print(f"WARNING: measured and learned amplitudes differ by more than 2x; "
                  f"check the open example really shows this session's door open.",
                  file=sys.stderr)
    else:
        print(f"rail amplitude: {amp:.1f} measured in this video")
    open_thr, close_thr = a.open_frac * amp, a.close_frac * amp
    occl_thr = max(4.0, a.occl_frac * amp)
    print(f"thresholds -> open>{open_thr:.1f} close<{close_thr:.1f} "
          f"occluded>{occl_thr:.1f}")

    occl = off_s > occl_thr
    state = np.zeros(len(on_s), np.int8)
    cur = 0
    for j in range(len(on_s)):
        if not occl[j]:                       # animal absent: update state
            cur = 1 if on_s[j] > open_thr else (0 if on_s[j] < close_thr else cur)
        state[j] = cur                        # animal present: hold
    lab, n = ndi.label(state == 1)
    for j in range(1, n + 1):
        w = np.nonzero(lab == j)[0]
        if len(w) < a.min_frames:
            state[w] = 0
    lab, n = ndi.label(state == 1)

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "on_rail", "surround", "occluded", "door_open"])
        for j in range(len(state)):
            w.writerow([j, round(float(on_c[j]), 2), round(float(off_t[j]), 2),
                        int(occl[j]), int(state[j])])
    stem = os.path.splitext(a.out)[0]
    with open(stem + "_intervals.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["interval", "start_frame", "end_frame", "n_frames",
                    "duration_s", "mean_on_rail", "frac_occluded"])
        for j in range(1, n + 1):
            q = np.nonzero(lab == j)[0]
            w.writerow([j, int(q[0]), int(q[-1]), len(q), round(len(q) / a.fps, 3),
                        round(float(on_c[q].mean()), 2), round(float(occl[q].mean()), 3)])
    print(f"wrote {a.out} and {stem}_intervals.csv: {len(state)} frames, "
          f"{n} open intervals, {state.mean()*100:.2f}% open, "
          f"{occl.mean()*100:.1f}% occluded by the animal")

    if a.qc:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(13, 8))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1])
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(ref, cmap="gray", vmin=0, vmax=255, extent=[x0, x1, y1, y0])
        ax.set_title("closed reference (this session)")
        ax2 = fig.add_subplot(gs[0, 1])
        if delta is not None:
            ax2.imshow(delta, cmap="inferno", vmin=0, vmax=max(6, 1.2 * amp),
                       extent=[x0, x1, y1, y0])
            ax2.set_title(f"closed − open; cyan = rail mask ({mask.sum()} px)")
        else:
            ax2.imshow(ref, cmap="gray", vmin=0, vmax=255, extent=[x0, x1, y1, y0])
            ax2.set_title(f"reused rail mask ({mask.sum()} px)")
        ax2.contour(mask, levels=[.5], colors="cyan", linewidths=.8,
                    extent=[x0, x1, y1, y0], origin="upper")
        for q in (ax, ax2):
            q.add_patch(plt.Rectangle((d["x"], d["y"]), d["w"], d["h"],
                                      ec="w", fc="none", lw=1))
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(on_c, lw=.3, color="tab:red", label="on-rail (drift removed)")
        ax3.plot(off_t, lw=.3, color="tab:blue", label="surround (animal)")
        ax3.axhline(open_thr, color="tab:red", ls=":", lw=.8)
        ax3.axhline(occl_thr, color="tab:blue", ls=":", lw=.8)
        ax3.set_ylim(-1.5 * amp, 3 * amp)
        ax3.legend(fontsize=8); ax3.set_ylabel("Δ vs closed reference")
        ax4 = fig.add_subplot(gs[2, :], sharex=ax3)
        ax4.fill_between(np.arange(len(state)), 0, state, step="mid",
                         color="tab:orange", lw=0)
        ax4.set_yticks([0, 1], ["closed", "open"])
        ax4.set_xlabel("behaviour video frame")
        ax4.set_title(f"{n} open intervals, {state.mean()*100:.2f}% of frames")
        fig.tight_layout(); fig.savefig(a.qc, dpi=130)
        print(f"wrote {a.qc}")


if __name__ == "__main__":
    main()
