#!/bin/bash
#SBATCH -p gengpu
#SBATCH -A p30771
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 24:00:00 # 1 hour in this example. gengpu has a max walltime of 48 hours.
#SBATCH --job-name="analyzevideos_deeplabcut_model_zoo"
#SBATCH --output /home/jma819/quest_deeplabcutscripts/logfiles/slurm.%x-%j.out 

module purge all
module load cuda/11.2.1-gcc-10.2.0
module load anaconda3

source activate tensorflow-2.6-py38-dlc

cd /home/jma819/quest_deeplabcutscripts


INPUT_project_name_to_create=$1

INPUT_behavCam_directory=$2

echo "creating project: $INPUT_project_name_to_create"

echo "directory with behavior videos is: $INPUT_behavCam_directory"

python dlc_analyze_new_videos_model_zoo.py $INPUT_project_name_to_create $INPUT_behavCam_directory 

