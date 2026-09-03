#!/bin/bash
#SBATCH --job-name=tis-model-evaluate-pretrained-transformer-fixwindow-clf
#SBATCH --output=tis-model-evaluate-pretrained-transformer-fixwindow-clf_%j.out
#SBATCH --partition=GPU
#SBATCH --ntasks=1

source $HOME/miniconda3/etc/profile.d/conda.sh

conda activate TIScalling

python script.py train 5 3
