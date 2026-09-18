#!/bin/bash

# slurm_dlc_finetune.sh — fine-tune SuperAnimal-TopViewMouse on blob-tracker labels.
#
# Same partition/account/env as slurm_superanimal_topview.sh (gengpu / p30771 /
# a100 / dlc3-torch), because this trains the same PyTorch HRNet top-down model.
#
# Before the first submission, on a login node (seconds, no GPU):
#   source activate dlc3-torch
#   python dlc_probe.py          # DLC version + the signatures this expects
#   python dlc_snapshots.py      # are the SuperAnimal snapshots on disk?
#   python dlc_snapshots.py --download    # if not; needs outbound network
# That prints the DLC version and the signatures dlc_finetune.py expects. If they
# differ, fix the script first -- an API mismatch discovered here costs a minute,
# discovered inside this job it costs the allocation.
#
# Build the project on the machine that holds the raw AVI (login node is fine).
# The selection manifest already exists; only the extraction needs the AVI, and
# step 2 takes the *_extracted.csv that step 1 writes, not the selection manifest:
#   python manifest_images.py --manifest C_trainframes_manifest.csv \
#       --from-video /scratch/jma819/T_maze_recordings/behaviorVIdeos/C.avi \
#       --outdir C_trainframes --geom maze_geometry_C.json
#   python dlc_project_from_blobs.py --manifest C_trainframes_manifest_extracted.csv \
#       --frames C_trainframes \
#       --from-h5 /scratch/jma819/T_maze_recordings/behaviorVIdeos/dlc_out/C_superanimal_topviewmouse_snapshot-hrnet_w32-004_snapshot-fasterrcnn_mobilenet_v3_large_fpn-004.h5 \
#       --video /scratch/jma819/T_maze_recordings/behaviorVIdeos/C.avi \
#       --project TmazeBlob --scorer blobtracker --out-root /projects/b1118/dlc_analysis
#
# Submit:
#   sbatch slurm_dlc_finetune.sh /projects/b1118/dlc_analysis/TmazeBlob-blobtracker-<date>/config.yaml
#   sbatch slurm_dlc_finetune.sh <config.yaml> /scratch/jma819/T_maze_recordings/behaviorVIdeos/C.avi

#SBATCH -p gengpu
#SBATCH -A p30771
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH --job-name="dlc_finetune"
#SBATCH --output /home/jma819/quest_deeplabcutscripts/logfiles/slurm.%x-%j.out

# Keep shell commands below the complete SBATCH block so Slurm reads every directive.
set -euo pipefail

module purge all
module load anaconda3

# `source activate` relies on conda shell functions that .bashrc defines on a
# login node. A batch job does not source .bashrc, which is why job 6535903 died
# in 11 s with ModuleNotFoundError while the same lines work interactively. Set
# the hook up explicitly, then fall back to the env's interpreter by path.
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

cd /home/jma819/quest_deeplabcutscripts

CONFIG=$1
ANALYZE=${2:-}
EPOCHS=${3:-100}

if [ -z "$CONFIG" ]; then
  echo "usage: sbatch slurm_dlc_finetune.sh <config.yaml> [video_to_analyze] [epochs]"
  exit 1
fi

# Snapshots must already be on disk: build_weight_init downloads them itself,
# but it does that inside this job, and a compute node may have no network.
if ! "$PY" dlc_snapshots.py; then
  echo "resolve the snapshots on a login node, then resubmit"
  exit 1
fi

echo "config  : $CONFIG"
echo "analyze : ${ANALYZE:-none}"
echo "epochs  : $EPOCHS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

ARGS=(--config "$CONFIG" --epochs "$EPOCHS" --batch-size 8)
if [ -n "$ANALYZE" ]; then
  ARGS+=(--analyze "$ANALYZE" --dest "$(dirname "$ANALYZE")/dlc_out_finetuned")
fi

"$PY" dlc_finetune.py "${ARGS[@]}"

echo "done"
