#!/bin/bash
# slurm_dlc_label_frames.sh -- extract a stratified set of frames for human
# labelling, one animal per array task.
#
# CPU only: this decodes a few hundred seeks out of a 250k-frame AVI and runs
# k-means on the tracked coordinates. No GPU, no deeplabcut import.
#
#   sbatch --array=0-2 slurm_dlc_label_frames.sh \
#     /scratch/jma819/T_maze_recordings/behaviorVIdeos \
#     /projects/b1118/dlc_analysis/TmazeParts-label \
#     C F K
#
# Writes, per animal X:
#   <project>/labeled-data/X/img*.png     the frames the GUI will show
#   <project>/X_labelframes.csv           frame, stratum, old x/y
#   <project>/X_labelframes.png           QC: where they came from
#
# The images are what you rsync to the laptop. Everything else is provenance.

#SBATCH -p normal
#SBATCH -A p30771
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH --job-name="dlc_label_frames"
#SBATCH --output /home/jma819/quest_deeplabcutscripts/logfiles/slurm.%x-%A_%a.out

module purge all
module load anaconda3

CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  . "$CONDA_BASE/etc/profile.d/conda.sh"
fi
conda activate dlc3-torch 2>/dev/null || source activate dlc3-torch 2>/dev/null
PY="$(command -v python)"
"$PY" -c "import cv2, scipy, pandas" || {
  echo "need cv2, scipy and pandas in this env"; exit 1; }

VIDDIR=$1; PROJ=$2; shift 2
ANIMALS=("$@")
if [ -z "$VIDDIR" ] || [ -z "$PROJ" ] || [ "${#ANIMALS[@]}" -eq 0 ]; then
  echo "usage: sbatch [--array=0-N] slurm_dlc_label_frames.sh <video_dir> <project_dir> <animal ...>"
  exit 1
fi
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  if [ "$SLURM_ARRAY_TASK_ID" -ge "${#ANIMALS[@]}" ]; then
    echo "array index beyond the animal list -- nothing to do"; exit 0
  fi
  ANIMALS=("${ANIMALS[$SLURM_ARRAY_TASK_ID]}")
fi

SUFFIX="DLC_HrnetW32_TmazeBlob2026-09-17shuffle1_detector_best-100_snapshot_best-10"
cd /home/jma819/quest_deeplabcutscripts

for A in "${ANIMALS[@]}"; do
  H5="$VIDDIR/dlc_out_finetuned/${A}${SUFFIX}.h5"
  VID="$VIDDIR/${A}.avi"
  for f in "$H5" "$VID"; do
    [ -f "$f" ] || { echo "missing $f -- skipping $A"; continue 2; }
  done
  echo "=== $A"
  "$PY" dlc_label_frames.py \
    --h5 "$H5" --video "$VID" \
    --outdir "$PROJ/labeled-data/$A" \
    --manifest "$PROJ/${A}_labelframes.csv" \
    --fig "$PROJ/${A}_labelframes.png" \
    --n "${NFRAMES:-200}" || exit 1
done

echo "done"
