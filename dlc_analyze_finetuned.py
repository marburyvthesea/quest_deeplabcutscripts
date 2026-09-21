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

WHAT THIS MODEL ACTUALLY IS (from the C training log, job 6679728)
------------------------------------------------------------------
Two networks are trained, and only one of them worked:

  DETECTOR (FasterRCNN mobilenet_v3_large_fpn) -- trained properly.
    test mAP@50:95 rose 42.10 -> 58.77 over 100 epochs, mAP@50 83.62, still
    improving at the end. Best snapshot = epoch 100 -> `detector_best-100`.

  POSE (HRNet w32 heatmap head) -- never trained.
    train loss 0.00026 at epoch 1, then 0.00023 flat for all 100 epochs.
    test.mAP 0.00 and rmse NaN at every one of the 10 evaluations. The loss is
    WeightedMSE over 27 heatmap channels of which 26 have no labels at all
    (only mouse_center is labelled), so at lr=1e-5 the gradient never moved it.

Consequences that matter when reading the output:

  * `snapshot_best-10` is a DEGENERATE TIE-BREAK, not a finding. The key metric
    (test.mAP) was 0.00 at all ten evaluations, so DLC kept the first. Every
    pose snapshot is the same essentially-unchanged SuperAnimal network.
  * The 26 non-centre keypoints are unusable. Per-bodypart evaluation reports
    NaN px for all of them; only mouse_center has a number.
  * Likelihoods carry no information: measured on C's 251,334 frames,
    mouse_center median 0.0028, 99th pct 0.0068, max 0.41. Nothing anywhere in
    the file reaches DLC's default pcutoff of 0.6, which is why the log's
    `rmse_pcutoff` is NaN.

What IS trustworthy is mouse_center's position, because the fine-tuned detector
localises the animal and the SuperAnimal pose prior places the centre inside
that box: train error 22.76 px, test error 40.86 px. That is coarse but ample
for arm occupancy and sensor-location work -- C's alignment validated to
p <= 0.008 on six held-out hardware channels using exactly these coordinates.

Validity is therefore the DETECTOR's verdict, not the pose confidence:
  valid frame   -> x > 0
  no detection  -> x == -1 and y == -1   (17% of C's frames)
This is what align_tmaze.py's SENTINEL check already keys on. Treat the 26
non-centre keypoints as unusable, not merely approximate.

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
    p.add_argument("--pcutoff", type=float, default=0.01,
                   help="deliberately low: this network's likelihoods are NOT "
                        "calibrated (see note below), so 0.6 would draw nothing")
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
