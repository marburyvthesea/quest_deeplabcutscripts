#!/bin/bash
# slurm_dlc_analyze_finetuned.sh — apply the trained TmazeBlob network to new videos.
#
# Inference only. Same partition/account/env as slurm_dlc_finetune.sh (gengpu /
# p30771 / a100 / dlc3-torch) because it is the same PyTorch top-down model; it
# just skips create_training_dataset and train_network.
#
# One video per array task, so F and K run concurrently on separate GPUs:
#
#   sbatch --array=0-1 slurm_dlc_analyze_finetuned.sh \
#     /projects/b1118/dlc_analysis/TmazeBlob-blobtracker-2026-09-17/config.yaml \
#     /scratch/jma819/T_maze_recordings/behaviorVIdeos/F.avi \
#     /scratch/jma819/T_maze_recordings/behaviorVIdeos/K.avi
#
# Without --array it analyses every video listed, serially, in one job.

#SBATCH -p gengpu
#SBATCH -A p30771
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --job-name="dlc_analyze_ft"
#SBATCH --output /home/jma819/quest_deeplabcutscripts/logfiles/slurm.%x-%A_%a.out

module purge all
module load anaconda3

# A batch job does not source .bashrc, so `source activate` has no conda shell
# functions to work with (this is what killed job 6535903 in 11 s). Set the hook
# up explicitly, then fall back to the env's interpreter by path.
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
echo "interpreter: $PY"
"$PY" -c "import deeplabcut, torch; print('deeplabcut', deeplabcut.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# Abort rather than fall back to CPU. analyze_videos does not refuse a missing
# GPU -- it just runs the top-down HRNet on CPU, which on a 250k-frame video
# means the job burns its entire wall clock and is killed with nothing written.
# A torch built for a newer CUDA than the node's driver shows up here as
# `cuda False` with "driver is too old", and that is fatal, not a warning.
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "torch cannot see a GPU on this node -- refusing to run on CPU."
  "$PY" -c "import torch; print('torch built for CUDA', torch.version.cuda)"
  nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
    || echo "nvidia-smi unavailable"
  echo "fix the torch/driver mismatch, then resubmit"
  exit 1
fi

cd /home/jma819/quest_deeplabcutscripts

CONFIG=$1
shift
if [ -z "$CONFIG" ] || [ "$#" -eq 0 ]; then
  echo "usage: sbatch [--array=0-N] slurm_dlc_analyze_finetuned.sh <config.yaml> <video> [video ...]"
  exit 1
fi

VIDEOS=("$@")
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  if [ "$SLURM_ARRAY_TASK_ID" -ge "${#VIDEOS[@]}" ]; then
    echo "array index $SLURM_ARRAY_TASK_ID >= ${#VIDEOS[@]} videos -- nothing to do"
    exit 0
  fi
  VIDEOS=("${VIDEOS[$SLURM_ARRAY_TASK_ID]}")
fi

echo "config : $CONFIG"
echo "videos : ${VIDEOS[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" dlc_analyze_finetuned.py --config "$CONFIG" --videos "${VIDEOS[@]}"

echo "done"
