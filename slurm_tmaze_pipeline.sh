#!/bin/bash
# slurm_tmaze_pipeline.sh -- door detection + behaviour/miniscope alignment,
# one animal per array task.
#
#   sbatch --array=0-1 slurm_tmaze_pipeline.sh F K
#   sbatch --array=0-2 slurm_tmaze_pipeline.sh C F K     # C re-run as a control
#
# THIS STAGE NEEDS NO GPU. It is video decode (OpenCV) plus numpy/pandas. The
# partition below is a CPU one; if `normal` is not the right queue for this
# allocation, change PART/ACCT -- gengpu/p30771 is known to work and will run
# it fine, just at the cost of holding an A100 for an hour.

#SBATCH -p normal
#SBATCH -A p30771
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH --job-name="tmaze_pipeline"
#SBATCH --output /home/jma819/quest_deeplabcutscripts/logfiles/slurm.%x-%A_%a.out

REPO=/home/jma819/quest_deeplabcutscripts
PIPELINE_ROOT=/scratch/jma819/T_maze_recordings/dataOrganizedForPipeline
VIDEO_DIR=/scratch/jma819/T_maze_recordings/behaviorVIdeos
TRACKING_DIR=$VIDEO_DIR/dlc_out_finetuned
OUTDIR=/scratch/jma819/T_maze_recordings/pipeline_out

module purge all
module load anaconda3
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  . "$CONDA_BASE/etc/profile.d/conda.sh"
fi
conda activate dlc3-torch 2>/dev/null || source activate dlc3-torch 2>/dev/null

PY="$(command -v python)"
"$PY" -c "import cv2" >/dev/null 2>&1 || PY=/home/jma819/.conda/envs/dlc3-torch/bin/python

# Fail here, with a readable message, rather than an hour into video decode.
if ! "$PY" - <<'EOF'
import importlib, sys
missing = [m for m in ("cv2", "numpy", "scipy", "pandas", "pyarrow", "tables", "matplotlib")
           if importlib.util.find_spec(m) is None]
if missing:
    print("missing modules:", missing)
    print("  pyarrow  -> needed to write *_traces_aligned.parquet")
    print("  tables   -> needed to read the DeepLabCut .h5")
    print("  install with: conda activate dlc3-torch && pip install " + " ".join(missing))
    sys.exit(1)
print("all required modules present")
EOF
then
  echo "environment incomplete -- fix and resubmit"
  exit 1
fi

cd "$REPO" || exit 1

ANIMALS=("$@")
if [ "${#ANIMALS[@]}" -eq 0 ]; then
  echo "usage: sbatch [--array=0-N] slurm_tmaze_pipeline.sh <animal> [animal ...]"
  exit 1
fi
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  if [ "$SLURM_ARRAY_TASK_ID" -ge "${#ANIMALS[@]}" ]; then
    echo "array index $SLURM_ARRAY_TASK_ID >= ${#ANIMALS[@]} animals -- nothing to do"
    exit 0
  fi
  ANIMALS=("${ANIMALS[$SLURM_ARRAY_TASK_ID]}")
fi

for A in "${ANIMALS[@]}"; do
  echo "================ $A"
  "$PY" run_tmaze_pipeline.py \
    --animal "$A" \
    --pipeline-root "$PIPELINE_ROOT" \
    --video-dir "$VIDEO_DIR" \
    --tracking-dir "$TRACKING_DIR" \
    --outdir "$OUTDIR" || exit 1
done

echo "done"
