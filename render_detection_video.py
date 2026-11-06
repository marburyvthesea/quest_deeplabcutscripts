"""Render a video that makes the detector's FAILURES visible, not just its hits.

deeplabcut.create_labeled_video draws a marker where there is a detection and
nothing where there is not, so the 78 % of frames with no detection look like
ordinary video -- the eye slides over them. This marks them explicitly: a red
border and a NO DETECTION banner, the blob tracker's position alongside for
comparison, and a running coverage counter so the rate is readable off the
screen.

  # 60 s segment starting at frame 60000 (fast, good enough to see the failure)
  python render_detection_video.py --animal C --start 60000 --n 1800

  # the whole video -- slow, large; expect ~2.3 h of footage for 250k frames
  python render_detection_video.py --animal C --n -1

Frames are read sequentially rather than sought, because seeking per frame on
a long compressed video is far slower and can land off-frame.
"""
import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd

V = "/scratch/jma819/T_maze_recordings/behaviorVIdeos"

DLC_C = (235, 111, 31)      # BGR: blue
BLOB_C = (91, 73, 209)      # BGR: red
BAD_C = (60, 60, 230)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--animal", required=True)
    p.add_argument("--video", default=None)
    p.add_argument("--h5", default=None)
    p.add_argument("--blob", default=None, help="blob_tracker CSV; omit to skip")
    p.add_argument("--bodypart", default="body_center")
    p.add_argument("--pcutoff", type=float, default=0.6)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=1800, help="-1 for the whole video")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    vid = a.video or f"{V}/{a.animal}.avi"
    h5 = a.h5 or glob.glob(f"{V}/dlc_out_finetuned/{a.animal}DLC_*.h5")[0]
    out = a.out or f"{a.animal}_detection.mp4"

    df = pd.read_hdf(h5)
    sub = df[df.columns.get_level_values(0)[0]]
    if sub.columns.nlevels == 3:
        sub = sub[sub.columns.get_level_values(0)[0]]
    X = sub[(a.bodypart, "x")].to_numpy(float)
    Y = sub[(a.bodypart, "y")].to_numpy(float)
    L = sub[(a.bodypart, "likelihood")].to_numpy(float)
    # the two encodings of "no detection" both have to count
    miss = ~np.isfinite(X) | ~np.isfinite(Y) | ((X == -1) & (Y == -1))

    bx = by = None
    if a.blob:
        B = pd.read_csv(a.blob)
        fcol = "frame" if "frame" in B.columns else B.columns[0]
        B = B.set_index(fcol)
        xc = "x" if "x" in B.columns else B.columns[0]
        yc = "y" if "y" in B.columns else B.columns[1]
        bx = B[xc].reindex(range(len(X))).to_numpy(float)
        by = B[yc].reindex(range(len(X))).to_numpy(float)

    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {vid}")
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    last = n_vid if a.n < 0 else min(n_vid, a.start + a.n)
    print(f"{os.path.basename(vid)}: {n_vid} frames, {w}x{h}; "
          f"rendering {a.start}..{last} stride {a.stride} -> {out}")
    print(f"no detection on {miss.mean()*100:.1f}% of the whole track")

    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"),
                         a.fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
    seen = hit = 0
    for fr in range(a.start, last):
        ok, im = cap.read()
        if not ok:
            break
        if (fr - a.start) % a.stride:
            continue
        seen += 1
        bad = fr >= len(miss) or miss[fr]
        if not bad:
            hit += 1
            x, y = int(round(X[fr])), int(round(Y[fr]))
            faint = L[fr] < a.pcutoff
            cv2.drawMarker(im, (x, y), DLC_C, cv2.MARKER_CROSS,
                           18, 1 if faint else 2)
            cv2.putText(im, f"p={L[fr]:.2f}", (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, DLC_C, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(im, (0, 0), (w - 1, h - 1), BAD_C, 6)
            cv2.putText(im, "NO DETECTION", (14, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, BAD_C, 2, cv2.LINE_AA)
        if bx is not None and fr < len(bx) and np.isfinite(bx[fr]):
            cv2.drawMarker(im, (int(round(bx[fr])), int(round(by[fr]))),
                           BLOB_C, cv2.MARKER_TILTED_CROSS, 16, 2)
        cv2.putText(im, f"frame {fr}   DLC coverage so far {100*hit/seen:4.1f}%",
                    (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (40, 40, 40), 2, cv2.LINE_AA)
        vw.write(im)
        if seen % 2000 == 0:
            print(f"  {fr}  coverage {100*hit/seen:.1f}%", flush=True)

    vw.release()
    cap.release()
    print(f"wrote {out}: {seen} frames, DLC present on {hit} "
          f"({100*hit/max(seen,1):.1f}%)")


if __name__ == "__main__":
    main()
