#!/bin/bash
#SBATCH --job-name=tis-model-training-gru-clf
#SBATCH --output=tis-model-training-gru-clf_%j.out
#SBATCH --partition=GPU
#SBATCH --ntasks=1

source $HOME/miniconda3/etc/profile.d/conda.sh

conda activate TIScalling

# Prepare PWM and codon usage table
python script/train_nn_model.py
