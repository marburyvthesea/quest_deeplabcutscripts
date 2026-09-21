#!/bin/bash
#SBATCH -p gengpu
#SBATCH -A p30771
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH --job-name="gpu_probe"
#SBATCH --output /home/jma819/quest_deeplabcutscripts/logfiles/slurm.%x-%j.out

module purge all
module load anaconda3
CONDA_BASE="$(conda info --base 2>/dev/null)"
[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] && . "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate dlc3-torch 2>/dev/null || source activate dlc3-torch 2>/dev/null
PY="$(command -v python)"
"$PY" -c "import deeplabcut; print('dlc', deeplabcut.__version__)" || \
  PY=/home/jma819/.conda/envs/dlc3-torch/bin/python

echo "--- node driver"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "--- torch"
"$PY" - <<'PYEOF'
import torch
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000, device="cuda")
    print("matmul on GPU ok:", float((x @ x).sum()) == float((x @ x).sum()))
PYEOF
