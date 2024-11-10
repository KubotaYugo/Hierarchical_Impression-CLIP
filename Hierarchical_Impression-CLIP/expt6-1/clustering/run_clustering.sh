#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for dataset in 'train' 'val' 'test'; do
    python programs/Hierarchical_Impression-CLIP/expt6-1/save_features_before_embedding/save_img_features.py --dataset ${dataset}
    for tag_preprocess in 'normal' 'average_single_tag' 'average_upto_10' 'single_tag'; do
        python programs/Hierarchical_Impression-CLIP/expt6-1/save_features_before_embedding/save_tag_features.py --dataset ${dataset} --tag_preprocess ${tag_preprocess}
    done
done

# bash programs/Hierarchical_Impression-CLIP/expt6-1/save_features_before_embedding/run_save_features.sh