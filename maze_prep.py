#!/usr/bin/env python3
"""maze_prep.py — one-time calibration + per-video preprocessing for T-maze behavior cams.

Two subcommands:

  calibrate  Interactive. Click the 4 maze corners, then trace a polygon that
             encloses ONLY the maze surface (the tracking ROI). Writes
             maze_calib.json holding the ROI polygon, the crop box, a
             pixel->cm homography, and a median reference image.

  prep       Non-interactive. Crops every video to the calibrated box, applies
             CLAHE to recover the saturated arena floor, and writes an mp4 that
             is what DeepLabCut and blob_tracker.py both consume.

Usage
-----
  python maze_prep.py calibrate VIDEO.avi --arena-w-cm 60 --arena-h-cm 40 -o maze_calib.json
  python maze_prep.py prep VIDEO_DIR -c maze_calib.json -o prepped/ [--ext .avi] [--no-clahe]

Deps: opencv-python, numpy.  Run `calibrate` on your laptop (needs a display);
`prep` runs headless and is the step to submit to the cluster for long videos.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------- utils


def sample_frames(path: Path, n: int = 120) -> np.ndarray:
    """Return n grayscale frames sampled evenly across the video."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"could not read frame count from {path}")
    idx = np.linspace(0, total - 1, min(n, total)).astype(int)
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not out:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(out)


def reference_image(frames: np.ndarray) -> np.ndarray:
    """Per-pixel median = arena with the animal and the swinging cable removed."""
    return np.median(frames, axis=0).astype(np.uint8)


def make_clahe(clip: float = 2.0, grid: int = 8) -> "cv2.CLAHE":
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))


# ------------------------------------------------------------------------- calibrate


def _click_collector(img: np.ndarray, title: str, n: int | None) -> list[tuple[int, int]]:
    """Collect clicks on img. n=None -> arbitrary many, finish with ENTER."""
    pts: list[tuple[int, int]] = []
    disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def draw():
        canvas = disp.copy()
        for i, p in enumerate(pts):
            cv2.circle(canvas, p, 4, (0, 0, 255), -1)
            cv2.putText(canvas, str(i + 1), (p[0] + 6, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        if len(pts) > 1:
            cv2.polylines(canvas, [np.array(pts)], n is not None, (0, 0, 255), 1)
        cv2.imshow(title, canvas)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and (n is None or len(pts) < n):
            pts.append((x, y))
            draw()
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()
            draw()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_click)
    draw()
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):                      # ENTER = done
            break
        if k == 27:                            # ESC = abort
            cv2.destroyAllWindows()
            sys.exit("aborted")
        if n is not None and len(pts) == n:
            cv2.waitKey(400)
            break
    cv2.destroyAllWindows()
    return pts


def cmd_calibrate(a: argparse.Namespace) -> None:
    frames = sample_frames(Path(a.video), a.n_ref)
    ref = reference_image(frames)
    view = make_clahe().apply(ref) if not a.no_clahe else ref

    print("\n[1/2] Click the 4 maze corners in order: "
          "stem-end-left, stem-end-right, arm-end-right, arm-end-left. "
          "Right-click undoes. ENTER when done.")
    corners = _click_collector(view, "maze corners (4)", 4)

    print("\n[2/2] Trace a polygon enclosing ONLY the maze surface. "
          "Keep the cable's off-maze excursion and the equipment rack OUTSIDE it. "
          "Right-click undoes, ENTER closes the polygon.")
    roi = _click_collector(view, "tracking ROI polygon", None)
    if len(roi) < 3:
        sys.exit("ROI needs >= 3 points")

    src = np.array(corners, np.float32)
    W, H = float(a.arena_w_cm), float(a.arena_h_cm)
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    homog = cv2.getPerspectiveTransform(src, dst)

    roi_arr = np.array(roi)
    m = int(a.margin)
    x0 = max(int(roi_arr[:, 0].min()) - m, 0)
    y0 = max(int(roi_arr[:, 1].min()) - m, 0)
    x1 = min(int(roi_arr[:, 0].max()) + m, ref.shape[1])
    y1 = min(int(roi_arr[:, 1].max()) + m, ref.shape[0])

    out = Path(a.out)
    calib = {
        "source_video": str(Path(a.video).resolve()),
        "frame_shape": [int(ref.shape[0]), int(ref.shape[1])],
        "corners_px": [[int(p[0]), int(p[1])] for p in corners],
        "arena_cm": [W, H],
        "homography_px_to_cm": homog.tolist(),
        "roi_polygon_px": [[int(p[0]), int(p[1])] for p in roi],
        "crop_box_px": [x0, y0, x1, y1],
        "clahe": (not a.no_clahe),
        "reference_image": out.with_suffix(".reference.png").name,
    }
    out.write_text(json.dumps(calib, indent=2))
    cv2.imwrite(str(out.with_suffix(".reference.png")), ref)

    overlay = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    cv2.polylines(overlay, [roi_arr.astype(np.int32)], True, (0, 0, 255), 2)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), 1)
    cv2.imwrite(str(out.with_suffix(".calib_overlay.png")), overlay)
    print(f"\nwrote {out}, {out.with_suffix('.reference.png')}, "
          f"{out.with_suffix('.calib_overlay.png')}")
    print(f"crop box {x0},{y0} -> {x1},{y1}  ({x1 - x0}x{y1 - y0} px)")


# ------------------------------------------------------------------------------ prep


def cmd_prep(a: argparse.Namespace) -> None:
    calib = json.loads(Path(a.calib).read_text())
    x0, y0, x1, y1 = calib["crop_box_px"]
    use_clahe = calib.get("clahe", True) and not a.no_clahe
    clahe = make_clahe()

    src = Path(a.videos)
    vids = sorted(src.rglob(f"*{a.ext}")) if src.is_dir() else [src]
    if not vids:
        sys.exit(f"no {a.ext} videos under {src}")
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    for v in vids:
        dst = outdir / f"{v.stem}_prepped.mp4"
        if dst.exists() and not a.overwrite:
            print(f"skip (exists) {dst.name}")
            continue
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = x1 - x0, y1 - y0
        wr = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), True)
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
            if use_clahe:
                g = clahe.apply(g)
            wr.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
            i += 1
            if i % 20000 == 0:
                print(f"  {v.name}: {i}/{n}", flush=True)
        cap.release()
        wr.release()
        print(f"wrote {dst.name}  ({i} frames, {fps:.2f} fps, {w}x{h})")


# ------------------------------------------------------------------------------ main


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate")
    c.add_argument("video")
    c.add_argument("-o", "--out", default="maze_calib.json")
    c.add_argument("--arena-w-cm", type=float, required=True)
    c.add_argument("--arena-h-cm", type=float, required=True)
    c.add_argument("--margin", type=int, default=15)
    c.add_argument("--n-ref", type=int, default=120)
    c.add_argument("--no-clahe", action="store_true")
    c.set_defaults(func=cmd_calibrate)

    q = sub.add_parser("prep")
    q.add_argument("videos")
    q.add_argument("-c", "--calib", default="maze_calib.json")
    q.add_argument("-o", "--out", default="prepped")
    q.add_argument("--ext", default=".avi")
    q.add_argument("--no-clahe", action="store_true")
    q.add_argument("--overwrite", action="store_true")
    q.set_defaults(func=cmd_prep)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
