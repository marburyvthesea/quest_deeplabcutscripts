#!/bin/bash
# slurm_blob_tracker.sh -- run blob_tracker.py over a whole session, one animal
# per array task, producing <animal>_blobtrack.csv.
#
# CPU only: this is a sequential decode of a 250k-frame AVI with connected
# components per frame. No GPU, no deeplabcut import.
#
#   sbatch --array=0-2 slurm_blob_tracker.sh \
#     /scratch/jma819/T_maze_recordings/behaviorVIdeos \
#     /scratch/jma819/T_maze_recordings/behaviorVIdeos \
#     C F K
#
# Writes, per animal X, into <outdir>:
#   X_mask.png / X_mask_bg.npy   surface mask QC and the cached background
#   X_blobtrack.csv              frame, x, y, area, depth, extent, elong,
#                                score, n_cand, status, interp, speed_px
#   X_blobtrack.png              occupancy and detection QC
#   maze_geometry_X.json         derived from the reference animal if absent
#
# The CSV is what dlc_label_frames.py --track consumes. Note the `interp`
# column: blob_tracker.py interpolates gaps up to --max-gap frames and flags
# them there. Rows with interp=True are the tracker's guess, NOT evidence the
# animal was visible, and the selector excludes them for that reason.
#
# WHY THE MASK STEP RUNS FIRST
# Two reasons. It caches the per-pixel background as X_mask_bg.npy, which the
# track step then reuses instead of re-seeking ~280 frames. And it is the only
# look you get at the surface mask, which is what --sat-thr and --erode control;
# if the mask is wrong every detection downstream is wrong in the same way.
# Check X_mask.png before trusting X_blobtrack.csv.
#
# ENVIRONMENT KNOBS
#   REF_ANIMAL=C        animal whose geometry/mask is the reference
#   GEOMDIR=<repo>      where maze_geometry_*.json live
#   TRACK_START, TRACK_N  tune on a window first (see TUNING below)
#   SAT_THR, ERODE, DARK_THR, AREA_MIN, AREA_MAX, MAX_STEP, MAX_GAP, FPS
#   FORCE=1             redo an animal whose CSV already exists
#
# TUNING
# Do not spend eight hours discovering the thresholds were wrong. Run a window
# with an overlay video first and watch it:
#
#   TRACK_START=5100 TRACK_N=500 OVERLAY=1 \
#     sbatch --array=0 slurm_blob_tracker.sh <viddir> <outdir> C
#
# That writes C_win5100.csv and C_win5100_overlay.mp4 in minutes. Only when the
# overlay looks right should you launch the full session.

#SBATCH -p normal
#SBATCH -A p30771
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 08:00:00
#SBATCH --job-name="blob_tracker"
#SBATCH --output /home/jma819/quest_deeplabcutscripts/logfiles/slurm.%x-%A_%a.out

set -euo pipefail

module purge all
module load anaconda3

# Batch shells do not reliably inherit interactive conda activation. Use the
# known environment directly instead of silently falling back to base Python.
PY=${PYTHON_BIN:-/home/jma819/.conda/envs/dlc3-torch/bin/python}
if [ ! -x "$PY" ]; then
  echo "Python interpreter not found or not executable: $PY"
  echo "Set PYTHON_BIN to the Python in an environment with the required packages."
  exit 1
fi
echo "Python: $PY"
if ! "$PY" - <<'PY'
import sys
from importlib import import_module

required = ("cv2", "matplotlib", "numpy", "pandas")
errors = {}
for name in required:
    try:
        import_module(name)
    except Exception as exc:
        errors[name] = f"{type(exc).__name__}: {exc}"
if errors:
    print("Python dependency check failed:", file=sys.stderr)
    for name, error in errors.items():
        print(f"  {name}: {error}", file=sys.stderr)
    print("Install them in dlc3-torch before resubmitting.", file=sys.stderr)
    raise SystemExit(1)
PY
then
  exit 1
fi

if [ "$#" -lt 3 ]; then
  echo "usage: sbatch [--array=0-N] $0 <video_dir> <out_dir> <animal ...>"
  exit 1
fi
VIDDIR=$1; OUTDIR=$2; shift 2
ANIMALS=("$@")
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  if [ "$SLURM_ARRAY_TASK_ID" -ge "${#ANIMALS[@]}" ]; then
    echo "array index beyond the animal list -- nothing to do"; exit 0
  fi
  ANIMALS=("${ANIMALS[$SLURM_ARRAY_TASK_ID]}")
fi

REPO=/home/jma819/quest_deeplabcutscripts
cd "$REPO"
mkdir -p "$OUTDIR"
GEOMDIR=${GEOMDIR:-$REPO}
REF_ANIMAL=${REF_ANIMAL:-C}
REF_GEOM="$GEOMDIR/maze_geometry_${REF_ANIMAL}.json"
REF_MASK="$GEOMDIR/DS0_rail_mask.json"

for A in "${ANIMALS[@]}"; do
  VID="$VIDDIR/${A}.avi"
  [ -f "$VID" ] || { echo "missing $VID -- skipping $A"; continue; }
  echo "=== $A"

  # ---- geometry. door_state.py and the rail mask were drawn on the reference
  # animal's pixels. Same rig, same day, so the furniture SHOULD sit in the same
  # place -- but a nudged camera makes every downstream number quietly wrong
  # rather than loudly broken, so measure the offset instead of assuming zero.
  GEOM="$GEOMDIR/maze_geometry_${A}.json"
  if [ ! -f "$GEOM" ]; then
    if [ "$A" = "$REF_ANIMAL" ]; then
      echo "no geometry for the reference animal $A at $GEOM -- draw it first"
      exit 1
    fi
    [ -f "$REF_GEOM" ] || { echo "no reference geometry at $REF_GEOM"; exit 1; }
    echo "deriving geometry for $A from $REF_ANIMAL"
    MASK_ARG=()
    [ -f "$REF_MASK" ] && MASK_ARG=(--ref-mask "$REF_MASK")
    "$PY" tmaze_geometry_transfer.py \
      --ref-video "$VIDDIR/${REF_ANIMAL}.avi" --ref-geom "$REF_GEOM" \
      "${MASK_ARG[@]}" \
      --video "$VID" --animal "$A" --outdir "$GEOMDIR" || exit 1
    [ -f "$GEOM" ] || { echo "transfer did not produce $GEOM"; exit 1; }
  fi

  COMMON=(--video "$VID" --geom "$GEOM"
          --sat-thr "${SAT_THR:-245}" --erode "${ERODE:-0}")

  # ---- surface mask + cached background
  MASK_PNG="$OUTDIR/${A}_mask.png"
  BG_NPY="$OUTDIR/${A}_mask_bg.npy"
  if [ ! -f "$BG_NPY" ] || [ -n "${FORCE:-}" ]; then
    "$PY" blob_tracker.py mask "${COMMON[@]}" --out "$MASK_PNG" || exit 1
  else
    echo "reusing $BG_NPY"
  fi

  # ---- track
  TRACK_ARGS=(--dark-thr "${DARK_THR:-200}"
              --area "${AREA_MIN:-200}" "${AREA_MAX:-3000}"
              --max-step "${MAX_STEP:-12}" --max-gap "${MAX_GAP:-15}"
              --fps "${FPS:-30}" --bg "$BG_NPY")
  if [ -n "${TRACK_N:-}" ]; then
    STEM="$OUTDIR/${A}_win${TRACK_START:-0}"
    TRACK_ARGS+=(--start "${TRACK_START:-0}" --n "$TRACK_N")
    [ -n "${OVERLAY:-}" ] && TRACK_ARGS+=(--overlay "${STEM}_overlay.mp4")
    echo "WINDOW RUN: frames ${TRACK_START:-0}..$(( ${TRACK_START:-0} + TRACK_N ))"
  else
    STEM="$OUTDIR/${A}_blobtrack"
    if [ -s "${STEM}.csv" ] && [ -z "${FORCE:-}" ]; then
      echo "${STEM}.csv exists -- set FORCE=1 to redo"; continue
    fi
  fi

  "$PY" blob_tracker.py track "${COMMON[@]}" "${TRACK_ARGS[@]}" \
    --out "${STEM}.csv" --fig "${STEM}.png" || exit 1

  # ---- what the selector will actually be able to use
  "$PY" - "${STEM}.csv" <<'PY' || exit 1
import sys
import numpy as np
import pandas as pd

T = pd.read_csv(sys.argv[1])
real = np.isfinite(T.x) & ~T.get("interp", pd.Series(False, index=T.index)).astype(bool)
print(f"  {len(T)} frames, {int(real.sum())} real detections "
      f"({100 * real.mean():.1f}%), "
      f"{int(T.get('interp', pd.Series(False)).astype(bool).sum())} gap-filled")
if real.mean() < 0.5:
    print("  WARNING: under half the frames tracked. Check the mask PNG and "
          "the --dark-thr/--area gates on a window before using this.")
PY
done

echo "done"
