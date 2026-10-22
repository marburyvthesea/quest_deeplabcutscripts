#!/bin/bash
# slurm_blob_diagnose.sh -- why did blob_tracker.py lose the animal on these
# frames? One animal per array task, reading an existing <animal>_blobtrack.csv
# and going back to the .avi to answer it.
#
#   sbatch --array=0-2 slurm_blob_diagnose.sh \
#     /scratch/jma819/T_maze_recordings/behaviorVIdeos \
#     /scratch/jma819/T_maze_recordings/behaviorVIdeos \
#     C F K
#
# CPU only, and much cheaper than the track it diagnoses: every step seeks to a
# few hundred frames rather than decoding the session. Minutes, not hours.
#
# Writes into <outdir>/diagnostics/ per animal X:
#   X_audit.csv    per failure run: cause, start, length, candidates available
#   X_sheet.png    contact sheet, one frame per run, labelled with its cause
#   X_probe.csv    nearest candidate vs the association gate, per miss frame
#   X_sweep.csv    darkness/depth grid: what each setting would recover, and
#                  what it costs in spurious candidates on frames that tracked
#   X_clips.mp4    the longest runs, with every candidate drawn and scored
#
# READ THE AUDIT FIRST. It splits the misses two ways and they need opposite
# fixes, so acting before reading it is a coin flip:
#
#   segmentation (n_cand=0)  nothing passed the darkness/area/depth gates. The
#                            animal was invisible to the detector -- occluded,
#                            under the door, or baked into the background
#                            model. Look at X_sheet.png and the sweep.
#   association  (n_cand>0)  a blob WAS found and the motion gate threw it out.
#                            Look at X_probe.csv: it reports how far the nearest
#                            candidate sat from the gate and which --max-step
#                            multiplier would admit it.
#
# Loosening --dark-thr to chase association loss adds false detections without
# recovering a frame; raising --max-step to chase segmentation loss does nothing
# at all. The sweep's second column exists to price the former: a setting that
# recovers misses while inflating candidates/frame on already-tracked frames is
# buying noise, not signal.
#
# THE GATE MUST MATCH THE TRACK
# probe and clips recompute the gate as --max-step x (gap+1). If the track was
# run with a non-default MAX_STEP, pass the SAME value here or the probe will
# report rejections against a gate the tracker never used:
#
#   MAX_STEP=20 sbatch --array=0 slurm_blob_diagnose.sh <viddir> <outdir> C
#
# ENVIRONMENT KNOBS
#   GEOMDIR=<repo>   where maze_geometry_*.json live
#   N_FRAMES=24      panels on the contact sheet
#   N_PROBE=400      miss frames probed
#   N_SWEEP=250      miss frames per grid point (and controls)
#   K_CLIPS=6        longest runs in the mp4; CLIPS=0 skips the video entirely
#   SAT_THR, ERODE, DARK_THR, AREA_MIN, AREA_MAX, MIN_DEPTH, MAX_STEP
#                    segmentation gates -- set them to the track's values
#   FORCE=1          redo an animal whose diagnostics already exist

#SBATCH -p normal
#SBATCH -A p30771
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 01:30:00
#SBATCH --job-name="blob_diagnose"
#SBATCH --output /home/jma819/quest_deeplabcutscripts/logfiles/slurm.%x-%A_%a.out

set -euo pipefail

module purge all
module load anaconda3

PY=${PYTHON_BIN:-/home/jma819/.conda/envs/dlc3-torch/bin/python}
if [ ! -x "$PY" ]; then
  echo "Python interpreter not found or not executable: $PY"
  echo "Set PYTHON_BIN to the Python in an environment with the required packages."
  exit 1
fi
echo "Python: $PY"

if [ "$#" -lt 3 ]; then
  echo "usage: sbatch [--array=0-N] $0 <video_dir> <track_dir> <animal ...>"
  exit 1
fi
VIDDIR=$1; OUTDIR=$2; shift 2
ANIMALS=("$@")

# Argument sanity. An unset shell variable in the submit line shifts every
# argument left; the job then fails as a "missing file" several steps later,
# which reads like a data problem and is not one. Check the shape here instead.
if [ ! -d "$VIDDIR" ] || ! compgen -G "$VIDDIR"/*.avi > /dev/null; then
  echo "no .avi files in '$VIDDIR' -- did an unset variable shift the arguments?"
  echo "usage: sbatch [--array=0-N] $0 <video_dir> <track_dir> <animal ...>"
  exit 1
fi
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  if [ "$SLURM_ARRAY_TASK_ID" -ge "${#ANIMALS[@]}" ]; then
    echo "array index beyond the animal list -- nothing to do"; exit 0
  fi
  ANIMALS=("${ANIMALS[$SLURM_ARRAY_TASK_ID]}")
fi

REPO=/home/jma819/quest_deeplabcutscripts
cd "$REPO"
GEOMDIR=${GEOMDIR:-$REPO}
DIAG="$OUTDIR/diagnostics"
mkdir -p "$DIAG"

for A in "${ANIMALS[@]}"; do
  VID="$VIDDIR/${A}.avi"
  TRACK="$OUTDIR/${A}_blobtrack.csv"
  GEOM="$GEOMDIR/maze_geometry_${A}.json"
  BG="$OUTDIR/${A}_mask_bg.npy"
  echo "=== $A"
  [ -f "$VID" ]   || { echo "missing $VID -- skipping $A"; continue; }
  [ -s "$TRACK" ] || { echo "no track at $TRACK -- run slurm_blob_tracker.sh for $A first"; continue; }
  [ -f "$GEOM" ]  || { echo "no geometry at $GEOM -- skipping $A"; continue; }

  if [ -s "$DIAG/${A}_probe.csv" ] && [ -z "${FORCE:-}" ]; then
    echo "diagnostics for $A exist -- set FORCE=1 to redo"; continue
  fi

  # Reuse the background the mask step cached. Without it every subcommand
  # re-seeks ~280 frames to rebuild the same array.
  BG_ARG=()
  if [ -f "$BG" ]; then BG_ARG=(--bg "$BG"); else echo "  no cached $BG -- rebuilding background per step"; fi

  SEG=(--geom "$GEOM" "${BG_ARG[@]}"
       --sat-thr "${SAT_THR:-245}" --erode "${ERODE:-0}"
       --dark-thr "${DARK_THR:-200}"
       --area "${AREA_MIN:-200}" "${AREA_MAX:-3000}"
       --min-depth "${MIN_DEPTH:-40}")

  # ---- cause split. Cheap, CSV only, and it decides what the rest means.
  "$PY" blob_tracker.py audit --track "$TRACK" \
    --out "$DIAG/${A}_audit.csv" --fig "$DIAG/${A}_audit.png" || exit 1

  # ---- what the failures actually look like
  "$PY" blob_diagnose.py frames --video "$VID" --track "$TRACK" \
    --audit "$DIAG/${A}_audit.csv" --n "${N_FRAMES:-24}" --cols 6 \
    --out "$DIAG/${A}_sheet.png" || exit 1

  # ---- association loss: was a real blob there, just outside the gate?
  "$PY" blob_diagnose.py probe --video "$VID" --track "$TRACK" \
    --audit "$DIAG/${A}_audit.csv" "${SEG[@]}" \
    --max-step "${MAX_STEP:-12}" --n "${N_PROBE:-400}" \
    --out "$DIAG/${A}_probe.csv" --fig "$DIAG/${A}_probe.png" || exit 1

  # ---- segmentation loss: would looser gates recover it, and at what cost?
  "$PY" blob_diagnose.py sweep --video "$VID" --track "$TRACK" \
    --audit "$DIAG/${A}_audit.csv" "${SEG[@]}" \
    --n "${N_SWEEP:-250}" --n-control "${N_SWEEP:-250}" \
    --out "$DIAG/${A}_sweep.csv" --fig "$DIAG/${A}_sweep.png" || exit 1

  # ---- eyes on the worst runs
  if [ "${CLIPS:-1}" != "0" ]; then
    "$PY" blob_diagnose.py clips --video "$VID" --track "$TRACK" \
      --audit "$DIAG/${A}_audit.csv" "${SEG[@]}" \
      --k "${K_CLIPS:-6}" --out "$DIAG/${A}_clips.mp4" || exit 1
  fi
done

echo "done"
