#!/bin/bash
# slurm_blob_calibrate.sh -- read blob_tracker.py thresholds off the video
# itself, one video per array task, before spending hours tracking with the
# wrong ones.
#
# CPU only, a few minutes per video: it decodes a few hundred sampled frames and
# re-runs the tracker's own gates over a parameter grid. No GPU, no deeplabcut.
#
#   sbatch --array=0-4 slurm_blob_calibrate.sh \
#     /scratch/jma819/T_maze_recordings/behaviorVIdeos \
#     /scratch/jma819/T_maze_recordings/behaviorVIdeos \
#     311_test 326_test 326_test_part2 328_test 388_test
#
# Writes, per video X, into <outdir>:
#   X_calib.csv   the parameter grid with a measured candidate rate per cell
#   X_calib.png   recommended mask, where the moving object was seen, animal vs
#                 background grey histograms, and the grid as a heatmap
#
# WHY THIS EXISTS
# The gates were tuned on the bright sessions (C, F, H, K), where the runnable
# surface is near-saturated. surface_mask keeps bg >= --sat-thr (default 245)
# and candidates computes (grey < --dark-thr) * mask, so a detection can only
# exist inside that mask. On a dimmer video the surface never reaches 245, the
# mask collapses to a few hundred saturated pixels of lamp or reflection, and
# the tracker reports 0 candidates per frame -- not weak detections, none. That
# is a mask failure wearing a threshold failure's clothes, and no amount of
# --dark-thr tuning touches it.
#
# This finds the animal by motion instead of by brightness, then reads the gates
# off the pixels it actually occupied, and tests them with the tracker's own
# functions. Read the RECOMMENDATION block in the log, then the PNG.
#
# ENVIRONMENT KNOBS
#   GEOMDIR=<repo>   where maze_geometry_*.json live
#   N=240            frames sampled for calibration
#   GRID_N=60        frames used per grid cell
#   WINDOW="5100 500"  calibrate on one window instead of the whole video
#   NOISE_K=4.0      motion threshold, in robust sigmas of frame-to-background
#   FORCE=1          redo a video whose _calib.csv already exists
#
# A geometry file is required and is NOT derived here: if maze_outer does not
# cover the runnable surface, the calibration will correctly report that nothing
# moves in the eligible region, which is a geometry problem and not a threshold
# problem. Draw or transfer the geometry first.

#SBATCH -p normal
#SBATCH -A p30771
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH --job-name="blob_calibrate"
#SBATCH --output /home/jma819/quest_deeplabcutscripts/logfiles/slurm.%x-%A_%a.out

set -euo pipefail

module purge all
module load anaconda3

REPO=${REPO:-/home/jma819/quest_deeplabcutscripts}
PY=${PYTHON_BIN:-/home/jma819/.conda/envs/dlc3-torch/bin/python}
cd "$REPO"
echo "Python: $PY"

if [ "$#" -lt 3 ]; then
  echo "usage: $0 <viddir> <outdir> <name> [name ...]"
  echo "  e.g. $0 /scratch/jma819/T_maze_recordings/behaviorVIdeos \\"
  echo "          /scratch/jma819/T_maze_recordings/behaviorVIdeos 311_test 326_test"
  exit 1
fi

VIDDIR=$1; OUTDIR=$2; shift 2
NAMES=("$@")

# the same guard as the other wrappers: an unset variable in the sbatch line
# silently shifts everything left, and the first symptom is a confusing error
# several steps downstream
if [ ! -d "$VIDDIR" ] || ! compgen -G "$VIDDIR"/*.avi > /dev/null; then
  echo "no .avi files in '$VIDDIR' -- did an unset variable shift the arguments?"
  exit 1
fi

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  if [ "$SLURM_ARRAY_TASK_ID" -ge "${#NAMES[@]}" ]; then
    echo "array index $SLURM_ARRAY_TASK_ID beyond the ${#NAMES[@]} names given -- nothing to do"
    exit 0
  fi
  NAMES=("${NAMES[$SLURM_ARRAY_TASK_ID]}")
fi

mkdir -p "$OUTDIR"
GEOMDIR=${GEOMDIR:-$REPO}

for A in "${NAMES[@]}"; do
  echo "=== $A"
  VID="$VIDDIR/${A}.avi"
  [ -f "$VID" ] || { echo "no video at $VID"; exit 1; }

  GEOM="$GEOMDIR/maze_geometry_${A}.json"
  if [ ! -f "$GEOM" ]; then
    echo "no geometry at $GEOM -- draw or transfer it before calibrating"
    exit 1
  fi

  OUT="$OUTDIR/${A}_calib.csv"
  if [ -f "$OUT" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "  $OUT exists; set FORCE=1 to redo"
    continue
  fi

  ARGS=(--video "$VID" --geom "$GEOM"
        --n "${N:-240}" --grid-n "${GRID_N:-60}"
        --noise-k "${NOISE_K:-4.0}"
        --out-prefix "$OUTDIR/${A}")

  # reuse the background the mask step cached, if it is there -- re-seeking a
  # few hundred frames of an 89k-frame AVI is the slowest part of this
  # NB: written as if-blocks, not `[ -f x ] && ARGS+=(...)`. Under `set -e` a
  # bare test-and-append aborts the whole script when the test is false, so a
  # missing background cache would present as a crash with no message.
  BG_NPY="$OUTDIR/${A}_mask_bg.npy"
  if [ -f "$BG_NPY" ]; then
    ARGS+=(--bg "$BG_NPY")
  fi

  if [ -n "${WINDOW:-}" ]; then
    # shellcheck disable=SC2206
    ARGS+=(--window ${WINDOW})
  fi

  "$PY" blob_calibrate.py "${ARGS[@]}" || exit 1
done

echo "done"
