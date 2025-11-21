#!/bin/bash

#SBATCH --job-name=Job_Recommender
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu002
#SBATCH --gres=gpu:4
#SBATCH --time=1-12:00:00
#SBATCH --output=job_output.out
#SBATCH --error=job_output.out

source $HOME/miniconda3/bin/activate
conda activate JobRecommender

# export CUDA_VISIBLE_DEVICES=4,5,6,7

CUDA_VISIBLE_DEVICES=4,5,6,7 srun --unbuffered torchrun --nproc_per_node=4 main.py --device 4,5,6,7 --model_type SplitBERT --version split_multigpu_overfit --batch_size 4 --epochs 100 --patience 100
# srun --unbuffered python main.py --device 0 --model_type SplitBERT --version split --batch_size 2