#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


for dataset in 'test' 'train'; do
    echo '----------------------------------------------------------------------------------------------------'
    echo "dataset=$dataset"
    python programs/Hierarchical_Impression-CLIP/expt6-3/data_preprocessing/font_preprocessor.py --dataset ${dataset}
done


# cd /media/user/0880c7a2-9e74-4110-a983-e8fd1fe49322/hierarchical-CLIP
# conda activate myenv
# bash programs/Hierarchical_Impression-CLIP/expt6-3/data_preprocessing/preprocess.sh