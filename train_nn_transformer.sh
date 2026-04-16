#!/bin/bash
#SBATCH --job-name=tis-model-training-transformer-clf
#SBATCH --output=tis-model-training-transformer-clf_%j.out
#SBATCH --partition=GPU
#SBATCH --ntasks=1

source $HOME/miniconda3/etc/profile.d/conda.sh

conda activate TIScalling

# Prepare PWM and codon usage table
python script/train_nn_model_by_site_transformer.py
