#!/bin/bash
#SBATCH -p gengpu
#SBATCH -A p30771
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 24:00:00 # 1 hour in this example. gengpu has a max walltime of 48 hours.
#SBATCH --job-name="deeplabcut_openfieldrun"
#SBATCH --output /home/jma819/quest_deeplabcutscripts/logfiles/slurm.%x-%j.out 

module purge all
module load cuda/cuda-9.2
module load anaconda3/2018.12

source activate tensorflow-gpu-env

cd /home/jma819/quest_deeplabcutscripts


INPUT_path_config_file=$1

INPUT_behavCam_directory=$2

echo "config file is: $INPUT_path_config_file"

echo "directory with behavior videos is: $INPUT_behavCam_directory"

python dlc_analyze_new_videos.py $INPUT_path_config_file $INPUT_behavCam_directory 

