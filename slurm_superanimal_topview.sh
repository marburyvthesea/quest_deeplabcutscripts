#!/bin/bash
# slurm_superanimal_topview.sh — SuperAnimal-TopViewMouse inference on Quest.
#
# Mirrors slurm_analyze_videos_model_zoo.sh (gengpu / p30771 / a100) but points at
# a PyTorch DeepLabCut 3.x environment instead of tensorflow-2.6-py38-dlc, because
# video adaptation and the HRNet top-down pose model are PyTorch-only.
#
# One-time environment build on a Quest login node:
#   module purge all
#   module load anaconda3
#   conda create -y -n dlc3-torch python=3.10
#   source activate dlc3-torch
#   pip install "deeplabcut[pytorch]" opencv-python-headless
#   python -c "import deeplabcut, torch; print(deeplabcut.__version__, torch.cuda.is_available())"
#
# Submit:
#   sbatch slurm_superanimal_topview.sh /projects/b1118/dlc_analysis/prepped adapt
#   sbatch slurm_superanimal_topview.sh /projects/b1118/dlc_analysis/prepped

#SBATCH -p gengpu
#SBATCH -A p30771
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH --job-name="superanimal_topview"
#SBATCH --output /home/jma819/quest_deeplabcutscripts/logfiles/slurm.%x-%j.out

module purge all
module load anaconda3
source activate dlc3-torch

cd /home/jma819/quest_deeplabcutscripts

VIDEO_DIR=$1
MODE=${2:-plain}
VIDEO_EXT=${3:-.avi}

# keep model-zoo weights on project storage rather than $HOME
export DLC_MODELZOO_CACHE=/projects/b1118/dlc_analysis/modelzoo_cache
mkdir -p "$DLC_MODELZOO_CACHE"

OUT=${VIDEO_DIR}/dlc_out
mkdir -p "$OUT"

echo "video dir : $VIDEO_DIR"
echo "mode      : $MODE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ "$MODE" == "adapt" ]; then
  # video adaptation on a single representative session; --limit 1 keeps it to one video
  python dlc_superanimal_topview.py "$VIDEO_DIR" \
      --ext "$VIDEO_EXT" --out "$OUT" --adapt --limit 1 \
      --pcutoff 0.15 --batch-size 8 --detector-batch-size 4 --max-individuals 1
else
  python dlc_superanimal_topview.py "$VIDEO_DIR" \
      --ext "$VIDEO_EXT" --out "$OUT" \
      --pcutoff 0.15 --batch-size 16 --detector-batch-size 8 --max-individuals 1 \
      --no-labeled-video
fi

echo "done"
