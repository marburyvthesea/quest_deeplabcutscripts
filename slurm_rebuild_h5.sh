#!/bin/bash
#SBATCH -p normal
#SBATCH -A p30771
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --job-name="rebuild_h5"
#SBATCH --output /home/jma819/quest_deeplabcutscripts/logfiles/slurm.%x-%j.out
#SBATCH --error  /home/jma819/quest_deeplabcutscripts/logfiles/slurm.%x-%j.err

# Rebuild the analysis h5 files from the _full.pickle using the correct
# SuperAnimal channels. This is pure file I/O -- no GPU, no inference, no
# model is loaded. The only reason this is a batch job rather than an
# interactive one is memory: the pickles are large and loading one is a
# multi-GB allocation that the login node will not tolerate.
#
# If 'normal' is not a partition you can submit to, check with
#   sinfo -s
# and change -p above. Any CPU partition works.

set -euo pipefail
module purge
eval "$(conda shell.bash hook)"
conda activate dlc3-torch

cd /home/jma819/quest_deeplabcutscripts
mkdir -p logfiles

python rebuild_h5_from_pickle.py "$@"
