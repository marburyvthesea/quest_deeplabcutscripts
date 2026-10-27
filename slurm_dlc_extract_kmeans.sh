#!/bin/bash
# Extract DLC k-means frames headlessly, one configured video per array task.
#
# NFRAMES=50 CLUSTER_STEP=10 sbatch --array=0-4 \
#   slurm_dlc_extract_kmeans.sh /path/to/project/config.yaml \
#   311_test 326_test 326_test_part2 328_test 388_test

#SBATCH -p normal
#SBATCH -A p30771
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH --job-name="dlc_extract_kmeans"
#SBATCH --output /home/jma819/quest_deeplabcutscripts/logfiles/slurm.%x-%A_%a.out

set -euo pipefail

module purge all
module load anaconda3

PY=${PYTHON_BIN:-/home/jma819/.conda/envs/dlc3-torch/bin/python}
if [ ! -x "$PY" ]; then
  echo "Python interpreter not found or not executable: $PY"
  exit 1
fi

if [ "$#" -lt 2 ]; then
  echo "usage: sbatch [--array=0-N] $0 <config.yaml> <video-stem ...>"
  exit 1
fi

CONFIG=$1
shift
STEMS=("$@")

if [ ! -f "$CONFIG" ]; then
  echo "config not found: $CONFIG"
  exit 1
fi

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  if [ "$SLURM_ARRAY_TASK_ID" -ge "${#STEMS[@]}" ]; then
    echo "array index beyond the video list -- nothing to do"
    exit 0
  fi
  STEMS=("${STEMS[$SLURM_ARRAY_TASK_ID]}")
fi

REPO=/home/jma819/quest_deeplabcutscripts
cd "$REPO"
mkdir -p logfiles

for STEM in "${STEMS[@]}"; do
  echo "=== Extracting frames for $STEM"
  MPLBACKEND=Agg DLClight=True "$PY" dlc_extract_kmeans_frames.py \
    --config "$CONFIG" \
    --video-stems "$STEM" \
    --num-frames "${NFRAMES:-50}" \
    --cluster-step "${CLUSTER_STEP:-10}" \
    --cluster-resizewidth "${CLUSTER_RESIZEWIDTH:-30}"
done

echo "done"
