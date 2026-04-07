#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=fat_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=1
#SBATCH --job-name=sherlock_quick_sanity
#SBATCH --account=melloldm_0000
#SBATCH --time 1:00:00


args="$@"
set --

conda activate boundarysegmentation

cd ~/projects/boundarysegmentation

python run_sherlock_sanity.sh $args