#!/bin/bash

#SBATCH --job-name=Job_Recommender
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu002
#SBATCH --gres=gpu:4
#SBATCH --time=1-12:00:00

source $HOME/miniconda3/bin/activate
conda activate JobRecommender

# export CUDA_VISIBLE_DEVICES=4,5,6,7

CUDA_VISIBLE_DEVICES=4,5,6,7 srun --unbuffered torchrun --nproc_per_node=4 main.py --device 4,5,6,7