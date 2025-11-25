#!/bin/bash

#SBATCH --job-name=SharedBERT
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu002
#SBATCH --gres=gpu:4
#SBATCH --time=1-12:00:00

source $HOME/miniconda3/bin/activate
conda activate JobRecommender

# export CUDA_VISIBLE_DEVICES=4,5,6,7

CUDA_VISIBLE_DEVICES=1,3,6,7 srun --unbuffered torchrun --master_port 29501 --nproc_per_node=4 main.py --device 1,3,6,7 --model_type FitClassifierBERT --version withTypeClassifierTransferCrossAttentionUnfrozen --batch_size 4 --learning_rate 1e-4



# CUDA_VISIBLE_DEVICES=0,1,2,3 srun --unbuffered torchrun --master_port 29502 --nproc_per_node=4 main.py --device 0,1,2,3 --model_type TypeClassifierBERT --version typeClassifier_20Continued --batch_size 64 --epochs 100
