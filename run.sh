#!/bin/bash

#SBATCH --job-name=Job_Recommender
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu001
#SBATCH --gres=gpu:4
#SBATCH --time=1-12:00:00

source $HOME/miniconda3/bin/activate
conda activate JobRecommender

# export CUDA_VISIBLE_DEVICES=4,5,6,7

# CUDA_VISIBLE_DEVICES=4,5,6,7 srun --unbuffered torchrun --master_port 29501 --nproc_per_node=4 main.py --device 4,5,6,7 --model_type SplitBERT --version split_multigpu_overfit --batch_size 4 --epochs 100 --patience 100
# CUDA_VISIBLE_DEVICES=0,1,2,3 srun --unbuffered torchrun --master_port 29502 --nproc_per_node=4 main.py --device 0,1,2,3 --model_type SplitBERT --version split_multigpu_overfit --batch_size 4 --epochs 100 --patience 100

# srun --unbuffered python main.py --device 0 --model_type SplitBERT --version split --batch_size 2

# for model 1(ML)
# srun --unbuffered python main.py --device 5 --model_type ML --version tfidf_baseline

# for model 3(Word2Vec)
srun --unbuffered python main.py --device 5 --model_type Word2Vec --version w2v_baseline
