#!/bin/bash
#SBATCH --job-name=tis-model-training-pretrained-transformer-clf
#SBATCH --output=tis-model-training-pretrained-transformer-clf_%j.out
#SBATCH --partition=GPU
#SBATCH --ntasks=1

source $HOME/miniconda3/etc/profile.d/conda.sh

conda activate TIScalling

# Prepare PWM and codon usage table
# python script/train_nn_model_by_site_rnafm_transformer_AIDORNA.py  # pretrain model AIDO.RNA 
python script/train_nn_model_by_site_rnafm_transformer_RiNALMo.py  # pretrain model AIDO.RNA 
