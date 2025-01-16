#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=<add jobname here>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16000
#SBATCH --time=24:00:00
#SBATCH --output=out/output.txt
#SBATCH --error=err/error.txt

# Add Miniconda to PATH
source $HOME/miniconda/etc/profile.d/conda.sh
conda init bash
conda activate # add name of conda env here

# Run the training script
python3 train_v2.py
