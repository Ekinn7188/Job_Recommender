#!/bin/bash

#SBATCH --job-name=SplitBERT
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu002
#SBATCH --gres=gpu:4
#SBATCH --time=1-12:00:00

source $HOME/miniconda3/bin/activate
conda activate JobRecommender

# export CUDA_VISIBLE_DEVICES=4,5,6,7

CUDA_VISIBLE_DEVICES=0,1,2,3 srun --unbuffered torchrun --master_port 29503 --nproc_per_node=4 main.py --device 0,1,2,3 --model_type SplitBERT --version split_latest --batch_size 4

# srun --unbuffered python main.py --device 0 --model_type SplitBERT --version split --batch_size 2
