#!/usr/bin/env python
"""
Extract per-frame intensity traces for candidate door regions over a whole video.

The hand-measured door ROI is a rough region, not the door, so instead of
committing to one signal this walks the video once and records the mean
intensity of every sub-cell of a grid tiling that rough region, plus the region
as a whole. A door that drops in ~1 s and returns leaves a stereotyped repeated
ramp in whichever sub-cells it actually covers; cells that only see the animal
or the cable do not repeat.

  python door_trace.py --video C.avi --geom maze_geometry_C.json \
         --door DS0 --pad 20 --grid 6 4 --out C_doortrace.csv
"""
import argparse
import json
import cv2
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--geom", required=True)
    p.add_argument("--door", default="DS0")
    p.add_argument("--pad", type=int, default=20, help="px to grow the rough ROI by")
    p.add_argument("--grid", type=int, nargs=2, default=[6, 4], metavar=("NX", "NY"))
    p.add_argument("--out", required=True)
    p.add_argument("--report-every", type=int, default=20000)
    a = p.parse_args()

    g = json.load(open(a.geom))
    b = g["doors"][a.door]
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    x0, x1 = max(0, b["x"] - a.pad), min(W, b["x"] + b["w"] + a.pad)
    y0, y1 = max(0, b["y"] - a.pad), min(H, b["y"] + b["h"] + a.pad)
    nx, ny = a.grid
    xe = np.linspace(x0, x1, nx + 1).astype(int)
    ye = np.linspace(y0, y1, ny + 1).astype(int)
    cols = [f"c{i}_{j}" for j in range(ny) for i in range(nx)]
    print(f"{a.video}: {W}x{H}, {N} frames; ROI x {x0}-{x1} y {y0}-{y1}; "
          f"{nx}x{ny} grid of ~{(x1-x0)//nx}x{(y1-y0)//ny} px cells", flush=True)

    rec = np.full((N, len(cols) + 1), np.nan, np.float32)
    k = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        gr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        patch = gr[y0:y1, x0:x1].astype(np.float32)
        v = [patch.mean()]
        for j in range(ny):
            for i in range(nx):
                v.append(patch[ye[j]-y0:ye[j+1]-y0, xe[i]-x0:xe[i+1]-x0].mean())
        if k < N:
            rec[k, :] = v
        k += 1
        if a.report_every and k % a.report_every == 0:
            print(f"  {k}/{N}", flush=True)
    cap.release()
    out = pd.DataFrame(rec[:k], columns=["roi_mean"] + cols)
    out.insert(0, "frame", np.arange(k))
    out.to_csv(a.out, index=False)
    print(f"decoded {k} frames (header said {N}); wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
