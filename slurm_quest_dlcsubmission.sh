#!/bin/bash
#SBATCH -p gengpu
#SBATCH -A <allocationID>
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00 # 1 hour in this example. gengpu has a max walltime of 48 hours.
#SBATCH --job-name="deeplabcut_openfieldrun"

module purge all
module load cuda/cuda-9.2
module load anaconda3/2018.12

source activate tensorflow-gpu-env

python run_dlc_openfield

