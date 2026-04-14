#!/bin/bash
#SBATCH --job-name=tis-model-training-reproduce
#SBATCH --output=tis-model-training-reproduce-getcwd-reproduce_%j.out
#SBATCH --partition=GPU
#SBATCH --ntasks=1

conda activate TIScalling

# Prepare PWM and codon usage table
python scripts/train_nn_model.py
