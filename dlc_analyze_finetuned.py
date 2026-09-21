#!/usr/bin/env python
"""Run the already-trained TmazeBlob network over new behaviour videos.

Inference only -- no training, no dataset creation. The network fine-tuned on
animal C's blob-tracker pseudo-labels is applied unchanged to other animals
recorded on the same rig, which is what produces an .h5 with the same scorer
name and the same 27 keypoints as C's:

  CDLC_HrnetW32_TmazeBlob2026-09-17shuffle1_detector_best-100_snapshot_best-10.h5
  ^                                        ^                ^
  video stem   project + shuffle           detector snapshot / pose snapshot

The snapshot suffix comes from `snapshotindex` and `detector_snapshotindex` in
the project config.yaml, NOT from anything passed here -- leave them alone and
new videos get the identical scorer string, so downstream code that hardcodes
it keeps working. This script prints the resolved scorer before it starts so a
mismatch is visible in the first ten lines of the log rather than at the end.

Use the raw AVI (C.avi, F.avi, K.avi). The *_fiji_mjpeg.avi transcodes exist
only so the videos can be scrubbed frame-by-frame in FIJI; re-encoding is not
needed for DLC and doubles the decode cost.

  python dlc_analyze_finetuned.py --config .../TmazeBlob-blobtracker-2026-09-17/config.yaml \
      --videos /scratch/jma819/T_maze_recordings/behaviorVIdeos/F.avi
"""
import argparse, inspect, os, sys


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="trained project's config.yaml")
    p.add_argument("--videos", nargs="+", required=True)
    p.add_argument("--dest", default=None,
                   help="output folder (default: <video dir>/dlc_out_finetuned)")
    p.add_argument("--shuffle", type=int, default=1)
    p.add_argument("--labeled-video", action="store_true",
                   help="also render a labelled mp4 (slow on a 250k-frame video)")
    p.add_argument("--pcutoff", type=float, default=0.6)
    a = p.parse_args()

    missing = [v for v in a.videos if not os.path.isfile(v)]
    if missing:
        sys.exit(f"videos not found: {missing}")
    cfg = os.path.abspath(a.config)
    if not os.path.isfile(cfg):
        sys.exit(f"config not found: {cfg}")

    import deeplabcut
    print(f"deeplabcut {deeplabcut.__version__}", flush=True)

    try:
        from deeplabcut.core.config import read_config_as_dict as _read
    except ImportError:
        from deeplabcut.utils.auxiliaryfunctions import read_plainconfig as _read
    c = _read(cfg)
    print(f"project   : {c.get('Task')}-{c.get('scorer')}-{c.get('date')}", flush=True)
    print(f"bodyparts : {len(c.get('bodyparts', []))}", flush=True)
    print(f"snapshotindex={c.get('snapshotindex')} "
          f"detector_snapshotindex={c.get('detector_snapshotindex')}", flush=True)

    dest = a.dest
    for v in a.videos:
        d = dest or os.path.join(os.path.dirname(os.path.abspath(v)),
                                 "dlc_out_finetuned")
        os.makedirs(d, exist_ok=True)
        print(f"\n=== {os.path.basename(v)} -> {d}", flush=True)
        kw = dict(shuffle=a.shuffle, destfolder=d, save_as_csv=True)
        sig = inspect.signature(deeplabcut.analyze_videos).parameters
        scorer = deeplabcut.analyze_videos(cfg, [v], **{k: w for k, w in kw.items()
                                                        if k in sig})
        print(f"scorer    : {scorer}", flush=True)
        if a.labeled_video:
            lkw = dict(shuffle=a.shuffle, destfolder=d, pcutoff=a.pcutoff)
            lsig = inspect.signature(deeplabcut.create_labeled_video).parameters
            try:
                deeplabcut.create_labeled_video(
                    cfg, [v], **{k: w for k, w in lkw.items() if k in lsig})
            except Exception as e:
                print(f"create_labeled_video failed ({type(e).__name__}: {e})",
                      flush=True)
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
