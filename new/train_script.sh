#!/bin/bash

#SBATCH --job-name=GAN                # create a short name for your job
#SBATCH --output=logs/log-%j.out      # Name of stdout output file (%j expands to jobId)
#SBATCH --error=logs/log-%j.err       # Name of stderr output file (%j expands to jobId)
#SBATCH --partition=gpu               # Partition name
#SBATCH --no-requeue                  # no requeue after node failure
#SBATCH --time=12:00:00               # total run time limit (HH:MM:SS)

#SBATCH --nodes=1                     # node count
#SBATCH --ntasks=8                    # total number of tasks across all nodes
#SBATCH --cpus-per-task=8             # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=0                       # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8                  # use all 8 GPUs

# Create directory for logging files (if not existing)
mkdir -p logs

# activate conda env
eval "$(conda shell.bash hook)"
conda activate /scratch/vihps/vihps20/envs/training

# Set python path
export PYTHONPATH=/home/vihps/vihps20/gan_dev:$PYTHONPATH

# debugging flags
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
#export OPENBLAS_NUM_THREADS=1
#export OMP_NUM_THREADS=1

srun python3 utils/run.py
