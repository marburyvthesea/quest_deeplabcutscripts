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

required = ("cv2", "matplotlib", "numpy", "pandas", "scipy", "tables")
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
  echo "usage: sbatch [--array=0-N] $0 <video_dir> <project_dir> <animal ...>"
  exit 1
fi
VIDDIR=$1; PROJ=$2; shift 2
ANIMALS=("$@")
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
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
