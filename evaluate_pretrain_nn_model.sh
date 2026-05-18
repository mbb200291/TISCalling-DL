#!/bin/bash
#SBATCH --job-name=tis-model-evaluate-pretrained-transformer-fixwindow-clf
#SBATCH --output=tis-model-evaluate-pretrained-transformer-fixwindow-clf_%j.out
#SBATCH --partition=GPU
#SBATCH --ntasks=1

source $HOME/miniconda3/etc/profile.d/conda.sh

conda activate TIScalling

# cd script/

# python script/evaluate_model.py rinalmo_dora_20260507_171051 model/rinalmo_dora_20260507_171051.pt
# python script/evaluate_model.py rinalmo_dora_20260508_095304 model/rinalmo_dora_20260508_095304.pt
# python script/evaluate_model.py rinalmo_dora_20260508_115620 model/rinalmo_dora_20260508_115620.pt
# python script/evaluate_model.py rinalmo_dora_20260513_010059 model/rinalmo_dora_20260513_010059.pt
# python script/evaluate_model.py rinalmo_dora_20260513_121752 model/rinalmo_dora_20260513_121752_bak.pt
# python script/evaluate_model.py rinalmo_dora_20260513_121752 model/rinalmo_dora_20260513_121752.pt
# python script/evaluate_model.py rinalmo_dora_20260513_132310 model/rinalmo_dora_20260513_132310.pt
# python script/evaluate_model.py rinalmo_dora_20260513_132026 model/rinalmo_dora_20260513_132026.pt
# python script/evaluate_model.py rinalmo_dora_20260507_170853 model/rinalmo_dora_20260507_170853.pt
# python script/evaluate_model.py rinalmo_dora_20260505_100930 model/rinalmo_dora_20260505_100930.pt
# python script/evaluate_model.py rinalmo_dora_20260505_132022 model/rinalmo_dora_20260505_132022.pt
# python script/evaluate_model.py rinalmo_dora_20260514_140724 model/rinalmo_dora_20260514_140724.pt
# python script/evaluate_model.py rinalmo_dora_20260514_114253 model/rinalmo_dora_20260514_114253.pt
# python script/evaluate_model.py rinalmo_dora_20260514_140724 model/rinalmo_dora_20260514_140724.pt
# python script/evaluate_model.py rinalmo_dora_20260515_090055 model/rinalmo_dora_20260515_090055.pt
# python script/evaluate_model.py rinalmo_dora_20260514_114253 model/rinalmo_dora_20260514_114253.pt
# python script/evaluate_model.py rinalmo_dora_20260515_134516 model/rinalmo_dora_20260515_134516.pt
python script/evaluate_model.py rinalmo_dora_20260515_134516 model/rinalmo_dora_20260515_134516.pt
