#!/bin/bash
#SBATCH --job-name=tiscalling-evaludate-model
#SBATCH --output=tiscalling-evaludate-model_%j.out
#SBATCH --partition=CPU
#SBATCH --ntasks=1

source $HOME/miniconda3/etc/profile.d/conda.sh

conda activate TIScalling

# Cross model evaluation
# python script/test_model.py -feature Features_Sl_5UTRATG.txt -model Model_Sl_5UTRATG,Model_Sl_CDSATG,Model_Sl_5UTRnonATG,Model_Sl_CDSnonATG
python script/test_model.py -feature Features_Sl_5UTRnonATG.txt -model Model_Sl_5UTRATG,Model_Sl_CDSATG,Model_Sl_5UTRnonATG,Model_Sl_CDSnonATG
python script/test_model.py -feature Features_Sl_CDSATG.txt -model Model_Sl_5UTRATG,Model_Sl_CDSATG,Model_Sl_5UTRnonATG,Model_Sl_CDSnonATG
python script/test_model.py -feature Features_Sl_CDSnonATG.txt -model Model_Sl_5UTRATG,Model_Sl_CDSATG,Model_Sl_5UTRnonATG,Model_Sl_CDSnonATG

