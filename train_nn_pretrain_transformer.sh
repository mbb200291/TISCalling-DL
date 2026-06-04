#!/bin/bash
#SBATCH --job-name=tis-model-training-pretrained-transformer-fixwindow-clf
#SBATCH --output=tis-model-training-pretrained-transformer-fixwindow-clf_%j.out
#SBATCH --partition=GPU
#SBATCH --ntasks=1

source $HOME/miniconda3/etc/profile.d/conda.sh

conda activate TIScalling

# Prepare PWM and codon usage table
# python script/train_nn_model_by_site_rnafm_transformer_AIDORNA.py  # pretrain model AIDO.RNA 
# python script/train_nn_model_by_site_rnafm_transformer_RiNALMo.py  # pretrain model AIDO.RNA 
# python script/train_nn_model_by_site_rnafm_transformer_RiNALMo_nonannotatedtis.py  # pretrain model AIDO.RNA and select non-canonical tis 
# python script/train_nn_model_by_site_rnafm_transformer_RiNALMo_nonannotatedtis_fixwindow.py  # pretrain model AIDO.RNA and select non-canonical tis, fixed window
# python script/train_nn_model_by_site_rnafm_transformer_RiNALMo_fixwindow.py  # pretrain model AIDO.RNA and fixed window
# python script/train_nn_model_by_site_rnafm_transformer_RiNALMo_nonannotatedtis_fixwindow_fixw.py  # pretrain model AIDO.RNA and fixed window
# python script/train_nn_model_by_site_rnafm_transformer_RiNALMo_nonannotatedtis_fixwindow_adjidv.py  # pretrain model AIDO.RNA and fixed window and reasonable wei adj
python script/rinalmo_dora_20260505_132022_grpperf.py
