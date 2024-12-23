#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for dataset in 'train' 'val' 'test'; do
    python programs/Hierarchical_Impression-CLIP/expt6-2-2/save_feature/save_img_feature.py --dataset ${dataset}
    for tag_preprocess in 'normal' 'average_single_tag' 'average_upto_10' 'single_tag'; do
        python programs/Hierarchical_Impression-CLIP/expt6-2-2/save_feature/save_tag_feature.py --dataset ${dataset} --tag_preprocess ${tag_preprocess}
    done
done

# bash programs/Hierarchical_Impression-CLIP/expt6-2-2/save_feature/run_save_feature.sh