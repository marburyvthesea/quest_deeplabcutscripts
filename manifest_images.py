#!/usr/bin/env python
"""
Materialise the training images named in a manifest.

Two sources, in order of preference:

  --from-dir  a directory of already-extracted raw frames, named frame_%05d.png
              (this is what DLC's video adaptation leaves in pseudo_<x>/images).
              Nothing is decoded, so this is fast and cannot drift from the
              frame indexing the track was built on.

  --from-video  decode the raw video and seek to each frame. Slower, and the
              script refuses a DLC-labeled overlay, which would bake the old
              markers into the training set.

Either way the image size is checked against the maze geometry the labels were
measured in, and the run aborts on a mismatch rather than producing silently
offset labels.

  python manifest_images.py --manifest C_trainframes_manifest.csv \
      --from-dir /scratch/jma819/T_maze_recordings/behaviorVIdeos/pseudo_C/images \
      --geom maze_geometry_C.json --outdir C_trainframes
"""
import argparse
import json
import os
import shutil

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--from-dir", default=None)
    p.add_argument("--from-video", default=None)
    p.add_argument("--geom", default=None, help="abort if image size disagrees with this")
    p.add_argument("--pattern", default="frame_{:05d}.png",
                   help="filename pattern in --from-dir")
    p.add_argument("--link", action="store_true",
                   help="symlink instead of copying (saves disk; breaks if the source moves)")
    p.add_argument("--allow-labeled", action="store_true")
    a = p.parse_args()
    if bool(a.from_dir) == bool(a.from_video):
        raise SystemExit("give exactly one of --from-dir or --from-video")

    M = pd.read_csv(a.manifest)
    os.makedirs(a.outdir, exist_ok=True)
    exp = None
    if a.geom:
        g = json.load(open(a.geom))
        exp = (g.get("width"), g.get("height"))
        if not all(exp):
            b = g["maze_outer"]
            print(f"geometry has no frame size; maze rectangle is "
                  f"x {b['x']}..{b['x']+b['w']}, y {b['y']}..{b['y']+b['h']} -- "
                  f"images must be at least that big")
            exp = None

    written, missing, size = [], [], None
    if a.from_dir:
        for fr in M.frame.astype(int):
            src = os.path.join(a.from_dir, a.pattern.format(int(fr)))
            if not os.path.exists(src):
                missing.append(int(fr))
                written.append("")
                continue
            dst = os.path.join(a.outdir, f"img{int(fr):07d}.png")
            if a.link:
                if os.path.lexists(dst):
                    os.remove(dst)
                os.symlink(os.path.abspath(src), dst)
            else:
                shutil.copy2(src, dst)
            written.append(os.path.basename(dst))
        probe = next((os.path.join(a.outdir, w) for w in written if w), None)
    else:
        import cv2
        if ("labeled" in os.path.basename(a.from_video).lower()
                and not a.allow_labeled):
            raise SystemExit(
                f"refusing {os.path.basename(a.from_video)}: the filename says it is a "
                "DLC overlay, so its frames carry the old markers. Point at the raw AVI.")
        cap = cv2.VideoCapture(a.from_video)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {a.from_video}")
        for fr in M.frame.astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
            ok, im = cap.read()
            if not ok:
                missing.append(int(fr))
                written.append("")
                continue
            nm = f"img{int(fr):07d}.png"
            cv2.imwrite(os.path.join(a.outdir, nm), im)
            written.append(nm)
        cap.release()
        probe = next((os.path.join(a.outdir, w) for w in written if w), None)

    if probe:
        import cv2
        im = cv2.imread(probe)
        size = (im.shape[1], im.shape[0])
        b = json.load(open(a.geom))["maze_outer"] if a.geom else None
        if b and (size[0] < b["x"] + b["w"] or size[1] < b["y"] + b["h"]):
            raise SystemExit(
                f"image size {size[0]}x{size[1]} is smaller than the maze rectangle "
                f"the labels were measured in (needs >= {b['x']+b['w']}x{b['y']+b['h']}). "
                "The labels would be offset. Check that these frames come from the same "
                "video as the track.")
        if exp and size != exp:
            raise SystemExit(f"image size {size} != geometry {exp}")

    M["image"] = written
    ok = M.loc[M.image != ""]
    out = os.path.splitext(a.manifest)[0] + "_extracted.csv"
    ok.to_csv(out, index=False)
    print(f"source: {a.from_dir or a.from_video}")
    print(f"image size: {size[0]}x{size[1]}" if size else "image size: unknown")
    print(f"wrote {len(ok)} of {len(M)} images to {a.outdir}/"
          f"{' (symlinks)' if a.link else ''}")
    if missing:
        print(f"MISSING {len(missing)} frames, e.g. {missing[:8]}")
    print(f"wrote {out}  <- pass this to dlc_project_from_blobs.py --manifest")


if __name__ == "__main__":
    main()
