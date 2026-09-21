#!/bin/bash
# Render labeled videos from existing fine-tuned DLC analysis outputs.
#
# This is CPU-only: create_labeled_video reads the existing H5/pickle files and
# encodes an overlay without running the detector or pose network again.
#
# Example (one video per array task, matching C's p60 output):
#   sbatch --array=0-1 slurm_dlc_create_labeled_finetuned.sh \
#     /projects/b1118/dlc_analysis/TmazeBlob-blobtracker-2026-09-17/config.yaml \
#     --pcutoff 0.6 \
#     /scratch/jma819/T_maze_recordings/behaviorVIdeos/F.avi \
#     /scratch/jma819/T_maze_recordings/behaviorVIdeos/K.avi

#SBATCH -p normal
#SBATCH -A p30771
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 08:00:00
#SBATCH --job-name="dlc_label_ft"
#SBATCH --output /home/jma819/quest_deeplabcutscripts/logfiles/slurm.%x-%A_%a.out

set -euo pipefail

module purge all
module load anaconda3

CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  . "$CONDA_BASE/etc/profile.d/conda.sh"
fi
conda activate dlc3-torch 2>/dev/null || source activate dlc3-torch 2>/dev/null

PY="$(command -v python)"
if ! "$PY" -c "import deeplabcut" >/dev/null 2>&1; then
  PY=/home/jma819/.conda/envs/dlc3-torch/bin/python
fi
if ! "$PY" -c "import deeplabcut" >/dev/null 2>&1; then
  echo "cannot import deeplabcut with '$PY' -- fix the environment, do not resubmit"
  exit 1
fi

cd /home/jma819/quest_deeplabcutscripts

CONFIG=${1:-}
if [ -z "$CONFIG" ]; then
  echo "usage: sbatch [--array=0-N] $0 <config.yaml> [--pcutoff VALUE] <video> [video ...]"
  exit 1
fi
shift

PCUTOFF=0.6
if [ "${1:-}" = "--pcutoff" ]; then
  if [ "$#" -lt 2 ]; then
    echo "--pcutoff requires a value"
    exit 1
  fi
  PCUTOFF=$2
  shift 2
fi

if [ "$#" -eq 0 ]; then
  echo "at least one video is required"
  exit 1
fi

VIDEOS=("$@")
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  if [ "$SLURM_ARRAY_TASK_ID" -ge "${#VIDEOS[@]}" ]; then
    echo "array index $SLURM_ARRAY_TASK_ID >= ${#VIDEOS[@]} videos -- nothing to do"
    exit 0
  fi
  VIDEOS=("${VIDEOS[$SLURM_ARRAY_TASK_ID]}")
fi

echo "config  : $CONFIG"
echo "videos  : ${VIDEOS[*]}"
echo "pcutoff : $PCUTOFF"

"$PY" dlc_analyze_finetuned.py \
  --config "$CONFIG" \
  --videos "${VIDEOS[@]}" \
  --label-only \
  --pcutoff "$PCUTOFF"

echo "done"
