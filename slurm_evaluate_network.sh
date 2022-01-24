#!/bin/bash
#SBATCH -p gengpu
#SBATCH -A p30771
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 04:00:00 # 1 hour in this example. gengpu has a max walltime of 48 hours.
#SBATCH --job-name="deeplabcut_openfieldrun"

module purge all
module load cuda/11.2.1-gcc-10.2.0
module load python/anaconda3

source activate tensorflow-2.6-py38-dlc


cd /home/jma819/quest_deeplabcutscripts


INPUT_path_config_file=$1



echo "config file is: $INPUT_path_config_file"



python dlc_evaluate_network.py $INPUT_path_config_file  

