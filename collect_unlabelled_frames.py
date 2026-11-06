"""Gather the extracted-but-never-labelled frames for re-annotation.

diag_bbox found the label set is bimodal: 292 frames with all five parts and
107 with none, nothing between. The empties are not inert. The 95/5 split was
computed over all 399 rows, so for a top-down model they can enter training
as images containing no object -- teaching the detector not to fire on frames
where the animal is plainly visible.

There are two dispositions and the contact sheet tells you which applies:

  the mouse IS visible   -> label them. 107 frames already extracted and on
                            disk, against ~600 from a fresh extraction round.
  the mouse is NOT visible (occluded, out of frame, experimenter's hand)
                         -> they were skipped on purpose. Labelling is
                            impossible, so drop the ROWS instead; leaving
                            them is what makes them negatives.

  python collect_unlabelled_frames.py                      # manifest + sheet
  python collect_unlabelled_frames.py --outdir resfiles    # + copy images
  python collect_unlabelled_frames.py --outdir resfiles --move

COPY is the default and --move is opt-in, because the images must stay under
labeled-data/<video>/ for DLC to associate them with CollectedData: move them
out and the GUI stops offering them and the rows point at missing files.
Nothing is ever deleted -- --prune-rows writes a CLEANED COPY of the
collected data beside the original and leaves the original untouched.
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd

P = "/projects/b1118/dlc_analysis/TmazeParts-jjm-2026-10-20"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=P)
    ap.add_argument("--outdir", default=None,
                    help="copy the unlabelled images here (omit to only "
                         "write the manifest and contact sheet)")
    ap.add_argument("--move", action="store_true",
                    help="move instead of copy -- breaks the DLC GUI "
                         "association, see module docstring")
    ap.add_argument("--prune-rows", action="store_true",
                    help="write a copy of the collected data with the empty "
                         "rows removed; never overwrites the original")
    ap.add_argument("--sheet", default="unlabelled_contact_sheet.png")
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--thumb", type=int, default=180)
    a = ap.parse_args()

    hits = sorted(glob.glob(
        f"{a.project}/training-datasets/*/*/CollectedData_*.h5"))
    if not hits:
        sys.exit(f"no merged CollectedData_*.h5 under {a.project}")
    merged = hits[0]
    D = pd.read_hdf(merged)
    empty = D.isna().all(axis=1).to_numpy()
    print(f"{os.path.basename(merged)}: {len(D)} rows, "
          f"{int(empty.sum())} with no labelled part")
    if not empty.any():
        sys.exit("nothing to do")

    rel = ["/".join(map(str, t)) if isinstance(t, tuple) else str(t)
           for t in D.index]
    M = pd.DataFrame({
        "relpath": [rel[i] for i in np.flatnonzero(empty)],
        "video": [rel[i].split("/")[-2] for i in np.flatnonzero(empty)],
        "image": [rel[i].split("/")[-1] for i in np.flatnonzero(empty)]})
    M["src"] = [os.path.join(a.project, r) for r in M.relpath]
    M["exists"] = [os.path.isfile(s) for s in M.src]
    print(M.video.value_counts().to_string())
    if not M.exists.all():
        print(f"\nWARNING: {int((~M.exists).sum())} image files are missing "
              "on disk; those rows can only be pruned, not labelled")

    M.to_csv("unlabelled_frames.csv", index=False)
    print(f"\nwrote unlabelled_frames.csv ({len(M)} rows)")

    if a.outdir:
        os.makedirs(a.outdir, exist_ok=True)
        if a.move:
            print("\n!! --move: DLC will no longer find these images under "
                  "labeled-data/, and the CollectedData rows will point at\n"
                  "   missing files. Only do this if you are re-extracting "
                  "or pruning the rows as well.")
        n = 0
        for _, r in M[M.exists].iterrows():
            # keep the video in the filename: img0001.png collides across videos
            dst = os.path.join(a.outdir, f"{r.video}__{r.image}")
            (shutil.move if a.move else shutil.copy2)(r.src, dst)
            n += 1
        print(f"{'moved' if a.move else 'copied'} {n} images -> {a.outdir}/")

    if a.prune_rows:
        out = merged.replace(".h5", "_pruned.h5")
        if os.path.exists(out):
            sys.exit(f"{out} exists; refusing to overwrite")
        D[~empty].to_hdf(out, key="df_with_missing", mode="w")
        print(f"wrote {out} ({int((~empty).sum())} rows, "
              f"{int(empty.sum())} empties removed) -- original untouched.\n"
              "  Swap it in only after create_training_dataset is re-run.")

    make_sheet(M[M.exists], a.sheet, a.cols, a.thumb)


def make_sheet(M, path, cols, thumb):
    """Tile the unlabelled frames so 'is the mouse visible?' is one glance."""
    if M.empty:
        print("no images on disk to tile")
        return
    import cv2
    rows = int(np.ceil(len(M) / cols))
    h = int(round(thumb * 2 / 3))
    sheet = np.full((rows * (h + 18), cols * thumb, 3), 255, np.uint8)
    for i, (_, r) in enumerate(M.iterrows()):
        im = cv2.imread(r.src)
        if im is None:
            continue
        im = cv2.resize(im, (thumb, h))
        y, x = (i // cols) * (h + 18), (i % cols) * thumb
        sheet[y:y + h, x:x + thumb] = im
        cv2.putText(sheet, f"{r.video} {r.image[:-4][-6:]}", (x + 2, y + h + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(path, sheet)
    print(f"wrote {path} ({len(M)} frames, {cols} per row)")


if __name__ == "__main__":
    main()
