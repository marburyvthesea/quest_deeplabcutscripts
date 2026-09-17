# T-maze position tracking with a large foreground cable

Pipeline to replace the ezTrack centroid track on the miniscope T-maze behavior
videos. Two independent estimators are run on the same preprocessed video and
their disagreement is the automatic QC signal.

## Diagnosis

Across the 10 sampled frames (frame 14,473 → 173,221, i.e. the whole session),
the reported centroid stayed inside a box covering **1.4 % of the camera field**,
always at x = 0.78–0.89 of field width — the far right edge, where the cable
leaves the maze. In the paired difference images, a **median 43 % of
supra-threshold difference pixels lie in the right 30 % of the field**, and the
horizontal centroid of the difference image sits at x = 0.56–0.61 of field width
in every frame. The track is not noisy; it is locked onto the cable.

The mechanism: ezTrack's location tracking thresholds `|frame − reference|` and
takes the intensity-weighted centroid. The commutator cable is **bright** and
moves over a **dark** surround, so it generates a larger absolute difference
than the animal, which is **dark** on a saturated-white arena floor. Because the
cable is larger in projected area than the mouse, no amount of smoothing
recovers the animal.

## Fix

Restrict the difference to **darkening only** (`reference − frame > thr`) inside
a maze ROI polygon. A bright object produces a positive excursion and is
discarded by construction rather than suppressed by a heuristic; permanently
dark structures inside the maze sit at the reference level and produce no
excursion at all. On a synthetic rig with the same geometry (bright sweeping
cable, dark animal, dark fixed features inside the arena), median error drops
from 47 px (absolute difference inside the ROI) to 1.5 px, with 80 % of frames
within 10 px of truth.

Residual failure mode: where the arena floor is locally dark, the animal has no
contrast against the reference and the estimate degrades (13.7 px median over a
dark feature vs 0.1 px elsewhere in the simulation). This is uncorrelated with
DeepLabCut's failure mode, which is why both are run.

## Steps

### 0. Cheap things to try first (minutes, no new tooling)

- In ezTrack's `LocationTracking` notebook, set the ROI to the maze surface only
  and enable the location-window option (`use_window=True`, `window_size` ≈ 2×
  body length, `window_weight` ≈ 0.9). This is ezTrack's own tether mitigation
  and was **not** evaluated here.
- Drop camera exposure/gain so the arena floor is no longer saturated. Every
  method below gains from the animal being separable from the floor.
- Sleeve the commutator cable in matte black heat-shrink. Against the dark
  surround it stops being a bright moving object at all.

### 1. Calibrate once per rig

```bash
python maze_prep.py calibrate m328/behavCam1.avi \
    --arena-w-cm 60 --arena-h-cm 40 -o m328/maze_calib.json
```

Click the 4 maze corners, then trace a polygon that encloses only the maze
surface. Writes `maze_calib.json` (ROI polygon, crop box, pixel→cm homography),
`maze_calib.reference.png` (median reference image) and an overlay to check.

### 2. Preprocess every video

```bash
python maze_prep.py prep m328/ -c m328/maze_calib.json -o m328/prepped --ext .avi
```

Crops to the maze, applies CLAHE, writes `*_prepped.mp4`. Both trackers consume
this file, so their coordinates are directly comparable.

### 3. Blob track (CPU, laptop or a single Quest core)

```bash
python blob_tracker.py m328/prepped/behavCam1_prepped.mp4 \
    -c m328/maze_calib.json -o m328/blob_track.csv \
    --clahe-ref --preview m328/blob_qc.mp4
```

Watch `blob_qc.mp4` for a few hundred frames and tune `--thr` (raise if shadows
are captured, lower if the animal is missed) and `--max-jump-px` (roughly
`max_speed_cm_s / fps` in pixels).

### 4. DeepLabCut SuperAnimal (Quest GPU)

```bash
# once, on a login node
conda create -y -n dlc3-torch python=3.10 && source activate dlc3-torch
pip install "deeplabcut[pytorch]" opencv-python-headless

# video adaptation on one representative session
sbatch slurm_superanimal_topview.sh /projects/b1118/dlc_analysis/m328/prepped adapt
# then the rest
sbatch slurm_superanimal_topview.sh /projects/b1118/dlc_analysis/m328/prepped
```

### 5. Merge and QC

```bash
python merge_qc.py m328/prepped/dlc_out/*.h5 m328/blob_track.csv \
    -c m328/maze_calib.json -o m328/merged_track.csv \
    --pcutoff 0.6 --fps 30 --agree-cm 3
```

Writes `merged_track.csv` with a `trust` column
(`both` / `conflict` / `dlc_only` / `blob_only` / `none`), a consensus position
in cm, and `tracking_qc.png`. Use `trust == "both"` frames for the manifold and
place-field analysis; exclude `conflict` and `none` frames rather than
interpolating across them.

## Notes on the DeepLabCut side

`create_pretrained_project(..., model="superanimal_topviewmouse")` in
`quest_deeplabcutscripts` routes to the **TensorFlow** DLCRNet weights
(`superanimal_topviewmouse_dlcrnet`). Video adaptation and the HRNet-w32
top-down model are **PyTorch-only**; engine routing in
`video_inference_superanimal` is by `model_name` (`dlcrnet` → TensorFlow,
anything else → PyTorch). Hence the separate `dlc3-torch` environment rather
than `tensorflow-2.6-py38-dlc`.

SuperAnimal-TopViewMouse predicts 27 keypoints: `nose`, `left_ear`,
`right_ear`, `left_ear_tip`, `right_ear_tip`, `left_eye`, `right_eye`, `neck`,
`mid_back`, `mouse_center`, `mid_backend`, `mid_backend2`, `mid_backend3`,
`tail_base`, `tail1`–`tail5`, `left_shoulder`, `left_midside`, `left_hip`,
`right_shoulder`, `right_midside`, `right_hip`, `tail_end`, `head_midpoint`.

Expect head keypoints (`nose`, ears, eyes) to be the weakest zero-shot: the
training corpus is overwhelmingly mice without a head-mounted miniscope and
without a bright tether crossing the head. Trunk keypoints (`mouse_center`,
`mid_back`, `tail_base`) should survive. `merge_qc.py` prints the per-keypoint
fraction of frames above `pcutoff` — that table decides the next move:

- Trunk keypoints > ~90 % and head keypoints > ~70 % after video adaptation:
  stop, you have what you need.
- Head keypoints still poor and you need head direction at the choice point:
  label 150–250 frames of your own video and fine-tune from the SuperAnimal
  weights. Stratify the labeled set: stem, choice point, both arms, reward
  zones, and specifically frames where the cable crosses the head.

## Untested

The four scripts are syntax-checked only. They have not been run against the
behavior videos — no behavior camera video was present in
`miniscope_analysis/miniscope_T_maze`, so the frame size, fps, and arena
dimensions in the examples above are placeholders. The synthetic validation in
`signed_difference_validation.png` exercises the signed-difference logic in
numpy, not `blob_tracker.py` itself.
