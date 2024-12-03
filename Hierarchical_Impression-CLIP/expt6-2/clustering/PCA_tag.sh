#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# for dataset in 'train' 'val' 'test'; do
for dataset in 'train'; do
    for tag_preprocess in 'normal' 'average_single_tag' 'average_upto_10'; do
        python programs/Hierarchical_Impression-CLIP/expt6-2/clustering/PCA_tag.py --dataset ${dataset} --tag_preprocess ${tag_preprocess}
        python programs/Hierarchical_Impression-CLIP/expt6-2/clustering/PCA_tag_number_of_tags.py --dataset ${dataset} --tag_preprocess ${tag_preprocess}
    done
done

# cd /media/user/0880c7a2-9e74-4110-a983-e8fd1fe49322/hierarchical-CLIP
# conda activate myenv
# bash programs/Hierarchical_Impression-CLIP/expt6-2/clustering/PCA_tag.sh