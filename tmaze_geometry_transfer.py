#!/usr/bin/env python
"""Carry a maze geometry (and a learned rail mask) from one animal's video to
another's, correcting for any camera shift between them.

WHY THIS EXISTS
---------------
door_state.py keys off pixel boxes: `doors.DS0` is an absolute ROI, and a saved
rail mask stores an absolute `box` plus a run-length mask inside it. Both were
drawn on C.avi. F and K were recorded on the same rig the same day, so the maze
furniture should sit in the same pixels -- but "should" is not "does", and if
the camera was nudged between sessions every downstream number is quietly wrong
rather than loudly broken. So we MEASURE the offset instead of assuming zero.

HOW
---
Take the per-pixel median of ~60 frames spread across each video. The mouse is
in a different place in every frame, so it averages out and what survives is
the static maze. Phase-correlate the two medians to get a sub-pixel translation,
then translate the geometry boxes and the mask box by that amount.

The sign convention of cv2.phaseCorrelate is calibrated at runtime against a
synthetic translation rather than taken from memory -- getting it backwards
would shift the ROI the wrong way by double the true offset and still "work".

`door_log_frames` is DROPPED on transfer: those are C's frame indices for
hand-identified door falls and mean nothing in another animal's video.

  python tmaze_geometry_transfer.py --ref-video C.avi --ref-geom maze_geometry_C.json \
      --ref-mask DS0_rail_mask.json --video F.avi --animal F --outdir .
"""
from __future__ import annotations
import argparse, json, os

import cv2
import numpy as np

MAX_ABS_SHIFT_PX = 60.0   # beyond this, refuse: the ROI should be redrawn
SAME_CAMERA_PX = 2.0      # within this, treat as unmoved


def reference_frame(video, n=60):
    """Per-pixel median of n frames spread over the video -> static scene."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    idx = np.linspace(0, max(N - 1, 0), n).astype(int)
    buf = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            buf.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not buf:
        raise SystemExit(f"no frames read from {video}")
    return np.median(np.stack(buf), axis=0).astype(np.float32), (W, H), N


def _phase_shift(a, b):
    """Raw cv2.phaseCorrelate on Hanning-windowed float images."""
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(np.ascontiguousarray(a),
                                        np.ascontiguousarray(b), win)
    return float(dx), float(dy), float(resp)


def _calibrate_sign(ref):
    """Translate ref by a known amount and see what phaseCorrelate reports.

    Returns +1 or -1 such that  true_shift = sign * reported_shift.
    """
    kx, ky = 7, -5
    M = np.float32([[1, 0, kx], [0, 1, ky]])
    moved = cv2.warpAffine(ref, M, (ref.shape[1], ref.shape[0]),
                           borderMode=cv2.BORDER_REFLECT)
    dx, dy, _ = _phase_shift(ref, moved)
    # pick the sign that reproduces the translation we actually applied
    return 1.0 if abs(dx - kx) + abs(dy - ky) <= abs(-dx - kx) + abs(-dy - ky) else -1.0


def estimate_shift(ref, tgt):
    """Translation (dx, dy) that maps REF coordinates onto TGT coordinates."""
    s = _calibrate_sign(ref)
    dx, dy, resp = _phase_shift(ref, tgt)
    return s * dx, s * dy, resp


def _shift_box(b, dx, dy):
    return {**b, "x": int(round(b["x"] + dx)), "y": int(round(b["y"] + dy))}


def shift_geometry(geom, dx, dy, animal):
    g = {k: v for k, v in geom.items() if k != "door_log_frames"}
    g["session"] = animal
    g["doors"] = {k: _shift_box(v, dx, dy) for k, v in geom["doors"].items()}
    if "maze_outer" in geom:
        g["maze_outer"] = _shift_box(geom["maze_outer"], dx, dy)
    if "exclude_islands" in geom:
        g["exclude_islands"] = [_shift_box(b, dx, dy)
                                for b in geom["exclude_islands"]]
    g["source"] = (f"transferred from {geom.get('session', '?')} by "
                   f"tmaze_geometry_transfer.py, shift ({dx:+.2f}, {dy:+.2f}) px")
    g["transfer"] = {"from": geom.get("session"), "dx_px": dx, "dy_px": dy}
    return g


def shift_mask(mj, dx, dy):
    """Translate a saved rail mask's box. The RLE inside is unchanged."""
    x0, x1, y0, y1 = mj["box"]
    ix, iy = int(round(dx)), int(round(dy))
    return {**mj, "box": [x0 + ix, x1 + ix, y0 + iy, y1 + iy],
            "transferred_by": [ix, iy]}


def qc_image(ref, tgt, geom_ref, geom_new, door, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    # one intensity scale for both, so a sampling difference in the medians is
    # not mistaken for a real change in the scene
    vmin = float(min(ref.min(), tgt.min())); vmax = float(max(ref.max(), tgt.max()))
    for ax, img, g, ttl in ((axes[0], ref, geom_ref, "reference"),
                            (axes[1], tgt, geom_new, "target")):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        d = g["doors"][door]
        ax.add_patch(plt.Rectangle((d["x"], d["y"]), d["w"], d["h"],
                                   fill=False, color="#d62728", lw=1.6))
        ax.set_title(f"{ttl}: {door} ROI", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def transfer(ref_video, ref_geom_path, video, animal, outdir,
             ref_mask_path=None, door="DS0", qc=True):
    """Returns (geom_path, mask_path_or_None, dx, dy, response)."""
    os.makedirs(outdir, exist_ok=True)
    geom = json.load(open(ref_geom_path))
    ref, ref_wh, _ = reference_frame(ref_video)
    tgt, tgt_wh, _ = reference_frame(video)
    if ref_wh != tgt_wh:
        raise SystemExit(f"frame sizes differ: reference {ref_wh} vs "
                         f"{animal} {tgt_wh}; geometry cannot be transferred")
    dx, dy, resp = estimate_shift(ref, tgt)
    mag = float(np.hypot(dx, dy))
    print(f"[{animal}] camera shift vs reference: dx={dx:+.2f} dy={dy:+.2f} "
          f"|d|={mag:.2f} px (phase-correlation response {resp:.3f})")
    if mag > MAX_ABS_SHIFT_PX:
        raise SystemExit(
            f"[{animal}] shift of {mag:.1f} px exceeds {MAX_ABS_SHIFT_PX:.0f} px. "
            "The camera moved too much to transfer ROIs; redraw the door box on "
            f"{os.path.basename(video)} and write maze_geometry_{animal}.json by hand.")
    if mag <= SAME_CAMERA_PX:
        print(f"[{animal}] within {SAME_CAMERA_PX:.0f} px -- camera treated as unmoved")
        dx = dy = 0.0

    gnew = shift_geometry(geom, dx, dy, animal)
    gpath = os.path.join(outdir, f"maze_geometry_{animal}.json")
    json.dump(gnew, open(gpath, "w"), indent=1)
    print(f"[{animal}] wrote {gpath}")

    mpath = None
    if ref_mask_path:
        mnew = shift_mask(json.load(open(ref_mask_path)), dx, dy)
        mpath = os.path.join(outdir, f"{door}_rail_mask_{animal}.json")
        json.dump(mnew, open(mpath, "w"), indent=1)
        print(f"[{animal}] wrote {mpath}")

    if qc:
        qpath = os.path.join(outdir, f"{animal}_geometry_qc.png")
        try:
            qc_image(ref, tgt, geom, gnew, door, qpath)
            print(f"[{animal}] wrote {qpath}  <- CHECK the red box is on the door")
        except Exception as e:
            print(f"[{animal}] QC image skipped ({type(e).__name__}: {e})")
    return gpath, mpath, dx, dy, resp


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref-video", required=True)
    p.add_argument("--ref-geom", required=True)
    p.add_argument("--ref-mask")
    p.add_argument("--video", required=True)
    p.add_argument("--animal", required=True)
    p.add_argument("--door", default="DS0")
    p.add_argument("--outdir", default=".")
    a = p.parse_args()
    transfer(a.ref_video, a.ref_geom, a.video, a.animal, a.outdir,
             ref_mask_path=a.ref_mask, door=a.door)


if __name__ == "__main__":
    main()
