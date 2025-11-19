#!/bin/bash

#SBATCH --job-name=Job_Recommender
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu002
#SBATCH --gres=gpu:8
#SBATCH --time=1-12:00:00

source $HOME/miniconda3/bin/activate
conda activate JobRecommender

srun --unbuffered python -u main.py --device 0,1,2,3,4,5,6,7