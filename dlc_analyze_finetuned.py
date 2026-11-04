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
import argparse, glob, inspect, os, sys


def install_single_animal_dataframe_fix():
    """Pad missing top-down detections before DLC 3.0.1 stacks frames."""
    import numpy as np
    from deeplabcut.pose_estimation_pytorch.apis import videos as video_api

    original = video_api.create_df_from_prediction

    def create_df_with_fixed_shape(*args, **kwargs):
        if args:
            raise TypeError("create_df_from_prediction must be called with keywords")

        predictions = kwargs["predictions"]
        model_cfg = kwargs["model_cfg"]
        multi_animal = kwargs["multi_animal"]
        if multi_animal:
            return original(**kwargs)

        metadata = model_cfg["metadata"]
        num_bodyparts = len(metadata["bodyparts"])
        num_individuals = max(1, len(metadata.get("individuals", [])))
        normalized = []
        missing = 0
        trimmed = 0

        for prediction in predictions:
            bodyparts = np.asarray(prediction["bodyparts"])
            if bodyparts.size == 0:
                bodyparts = np.empty((0, num_bodyparts, 3))
            if bodyparts.ndim != 3:
                raise ValueError(
                    f"unexpected bodypart prediction shape {bodyparts.shape}"
                )

            fixed = np.full((num_individuals, num_bodyparts, 3), np.nan)
            keep_individuals = min(num_individuals, bodyparts.shape[0])
            keep_bodyparts = min(num_bodyparts, bodyparts.shape[1])
            keep_coords = min(3, bodyparts.shape[2])
            if keep_individuals:
                fixed[:keep_individuals, :keep_bodyparts, :keep_coords] = (
                    bodyparts[:keep_individuals, :keep_bodyparts, :keep_coords]
                )
            else:
                missing += 1
            if bodyparts.shape[0] > num_individuals:
                trimmed += 1

            updated = dict(prediction)
            updated["bodyparts"] = fixed
            normalized.append(updated)

        if missing or trimmed:
            print(
                "normalized variable-size DLC predictions: "
                f"{missing} frames without detections, "
                f"{trimmed} frames trimmed to {num_individuals} individual(s)",
                flush=True,
            )
        kwargs["predictions"] = normalized
        return original(**kwargs)

    video_api.create_df_from_prediction = create_df_with_fixed_shape
    return video_api


def recover_dataframe_from_full_pickle(full_path, meta_path, video_api):
    """Create single-animal H5/CSV outputs from DLC's saved raw predictions."""
    import pickle
    import numpy as np

    with open(full_path, "rb") as stream:
        raw = pickle.load(stream)
    with open(meta_path, "rb") as stream:
        metadata = pickle.load(stream)["data"]

    bodyparts = list(metadata["bodyparts"])
    unique_bodyparts = list(metadata.get("unique_bodyparts", []))
    num_frames = raw["metadata"]["nframes"]
    key_width = raw["metadata"]["key_str_width"]
    predictions = []

    for frame_index in range(num_frames):
        frame = raw[f"frame{frame_index:0{key_width}d}"]
        coordinates = frame["coordinates"][0]
        confidence = frame["confidence"]
        xy = np.stack(coordinates[:len(bodyparts)], axis=0)
        scores = np.stack(confidence[:len(bodyparts)], axis=0)
        pose = np.concatenate((xy, scores), axis=-1).transpose((1, 0, 2))
        prediction = {"bodyparts": pose}

        if unique_bodyparts:
            first = len(bodyparts)
            last = first + len(unique_bodyparts)
            unique_xy = np.stack(coordinates[first:last], axis=0)
            unique_scores = np.stack(confidence[first:last], axis=0)
            prediction["unique_bodyparts"] = np.concatenate(
                (unique_xy, unique_scores), axis=-1
            ).transpose((1, 0, 2))
        predictions.append(prediction)

    model_cfg = {
        "metadata": {
            "bodyparts": bodyparts,
            "unique_bodyparts": unique_bodyparts,
            "individuals": metadata.get("individuals") or ["individual1"],
        }
    }
    prefix = os.path.basename(full_path).removesuffix("_full.pickle")
    video_api.create_df_from_prediction(
        predictions=predictions,
        dlc_scorer=metadata["Scorer"],
        multi_animal=False,
        model_cfg=model_cfg,
        output_path=os.path.dirname(full_path),
        output_prefix=prefix,
        save_as_csv=True,
    )
    return os.path.join(os.path.dirname(full_path), prefix + ".h5")


def find_labeled_videos(video, destfolder):
    """Return nonempty labeled videos matching one source video."""
    stem = os.path.splitext(os.path.basename(video))[0]
    pattern = os.path.join(destfolder, glob.escape(stem) + "*labeled.mp4")
    return sorted(path for path in glob.glob(pattern)
                  if os.path.isfile(path) and os.path.getsize(path) > 0)


def validate_labeled_video_result(video, destfolder, result):
    """Validate DLC's inconsistent return value against the actual output."""
    outputs = find_labeled_videos(video, destfolder)
    failed = result is False or (
        isinstance(result, (list, tuple))
        and (not result or any(item is False for item in result))
    )
    if failed or not outputs:
        raise RuntimeError(
            f"labeled video was not created: result={result}, "
            f"matching_outputs={outputs}")
    return outputs


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
    p.add_argument("--label-only", action="store_true",
                   help="render from existing analysis files without rerunning inference")
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
    video_api = install_single_animal_dataframe_fix()

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
        if not a.label_only:
            kw = dict(shuffle=a.shuffle, destfolder=d, save_as_csv=True)
            sig = inspect.signature(deeplabcut.analyze_videos).parameters
            scorer = deeplabcut.analyze_videos(
                cfg, [v], **{k: w for k, w in kw.items() if k in sig}
            )
            print(f"scorer    : {scorer}", flush=True)

            prefix = os.path.splitext(os.path.basename(v))[0] + scorer
            h5_path = os.path.join(d, prefix + ".h5")
            if not os.path.isfile(h5_path):
                full_path = os.path.join(d, prefix + "_full.pickle")
                meta_path = os.path.join(d, prefix + "_meta.pickle")
                if not (os.path.isfile(full_path) and os.path.isfile(meta_path)):
                    raise RuntimeError(
                        "DLC created neither the expected H5 nor recoverable "
                        f"pickle outputs for {v}"
                    )
                print(
                    "H5 missing but raw predictions exist; recovering without "
                    "rerunning inference",
                    flush=True,
                )
                h5_path = recover_dataframe_from_full_pickle(
                    full_path, meta_path, video_api
                )
            if not os.path.isfile(h5_path) or os.path.getsize(h5_path) == 0:
                raise RuntimeError(f"analysis output is missing or empty: {h5_path}")
            print(f"analysis output: {h5_path}", flush=True)
        else:
            print("analysis  : skipped (using existing outputs)", flush=True)

        if a.labeled_video or a.label_only:
            lkw = dict(shuffle=a.shuffle, destfolder=d, pcutoff=a.pcutoff)
            lsig = inspect.signature(deeplabcut.create_labeled_video).parameters
            try:
                created = deeplabcut.create_labeled_video(
                    cfg, [v], **{k: w for k, w in lkw.items() if k in lsig})
                outputs = validate_labeled_video_result(v, d, created)
                if isinstance(created, (list, tuple)) and any(
                        result is None for result in created):
                    print("DLC skipped an existing labeled video; verified output:",
                          flush=True)
                else:
                    print("labeled video output:", flush=True)
                for output in outputs:
                    print(f"  {output}", flush=True)
            except Exception as e:
                print(f"create_labeled_video failed ({type(e).__name__}: {e})",
                      flush=True)
                if a.label_only:
                    raise
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
