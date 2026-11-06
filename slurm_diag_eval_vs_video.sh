#!/bin/bash
# slurm_diag_eval_vs_video.sh — run evaluate_network and compare it to the
# analyze_videos output on identical frames. Same partition/account/env as the
# fine-tune, because it loads the same PyTorch top-down model.
#
#   sbatch slurm_diag_eval_vs_video.sh \
#     /projects/b1118/dlc_analysis/TmazeParts-jjm-2026-10-20/config.yaml

#SBATCH -p gengpu
#SBATCH -A p30771
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --job-name="dlc_eval_vs_video"
#SBATCH -o eval_vs_video_%j.out
#SBATCH -e eval_vs_video_%j.err

set -euo pipefail
CONFIG="${1:?usage: sbatch slurm_diag_eval_vs_video.sh /path/to/config.yaml}"

module purge
eval "$(conda shell.bash hook)"
conda activate dlc3-torch

cd "$SLURM_SUBMIT_DIR"
python diag_eval_vs_video.py --config "$CONFIG"
