#!/bin/bash

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

python3 utils/run.py "1.params"
python3 utils/run.py "5.params"
