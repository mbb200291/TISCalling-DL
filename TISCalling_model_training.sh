#!/bin/bash
#SBATCH --job-name=tis-model-training-reproduce
#SBATCH --output=tis-model-training-reproduce-grp-aware_%j.out
#SBATCH --partition=CPU
#SBATCH --ntasks=1

source $HOME/miniconda3/etc/profile.d/conda.sh

conda activate TIScalling

# # Prepare PWM and codon usage table
# python script/coden_usage_generator.py -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -seq input_data/Tomato_seq.txt -TP_type 2 -filename TIS_codon_Sl_5UTRATG.txt -sp Sl
# python script/coden_usage_generator.py -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -seq input_data/Tomato_seq.txt -TP_type 3 -filename TIS_codon_Sl_CDSATG.txt -sp Sl
# python script/coden_usage_generator.py -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -seq input_data/Tomato_seq.txt -TP_type 4 -filename TIS_codon_Sl_5UTRnonATG.txt -sp Sl
# python script/coden_usage_generator.py -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -seq input_data/Tomato_seq.txt -TP_type 5 -filename TIS_codon_Sl_CDSnonATG.txt -sp Sl

# python script/PWM_generator.py -seq input_data/Tomato_seq.txt -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -filename PWM_Sl_5UTRATG.txt -TP_type 2
# python script/PWM_generator.py -seq input_data/Tomato_seq.txt -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -filename PWM_Sl_CDSATG.txt -TP_type 3
# python script/PWM_generator.py -seq input_data/Tomato_seq.txt -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -filename PWM_Sl_5UTRnonATG.txt -TP_type 4
# python script/PWM_generator.py -seq input_data/Tomato_seq.txt -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -filename PWM_Sl_CDSnonATG.txt -TP_type 5

# # Generate features
# python script/get_features.py -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -seq input_data/Tomato_seq.txt -TIS_sp Sl -TIS_code 2
# python script/get_features.py -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -seq input_data/Tomato_seq.txt -TIS_sp Sl -TIS_code 3
# python script/get_features.py -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -seq input_data/Tomato_seq.txt -TIS_sp Sl -TIS_code 4
# python script/get_features.py -TP_TN_data input_data/Tomato_TP_and_TN_data.txt -seq input_data/Tomato_seq.txt -TIS_sp Sl -TIS_code 5

# Model training
python script/train_model.py -feature Features_Sl_CDSATG.txt
python script/train_model.py -feature Features_Sl_5UTRATG.txt
python script/train_model.py -feature Features_Sl_CDSnonATG.txt
python script/train_model.py -feature Features_Sl_5UTRnonATG.txt

# # Cross model evaluation
# python script/test_model.py -feature Features_Sl_5UTRATG.txt -model Model_Sl_5UTRATG,Model_Sl_CDSATG,Model_Sl_5UTRnonATG,Model_Sl_CDSnonATG

# # TIS prediction
# python script/predict_TIS.py -fasta input_data/predict_input.fa -models Model_Sl_5UTRATG,Model_Sl_CDSATG,Model_Sl_5UTRnonATG,Model_Sl_CDSnonATG

