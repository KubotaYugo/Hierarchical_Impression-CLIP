#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for dataset in 'train' 'val' 'test'; do
    for tag_preprocess in 'single_tag'; do
        # python programs/Hierarchical_Impression-CLIP/expt6-2/clustering/tag_feature_norm.py --dataset ${dataset} --tag_preprocess ${tag_preprocess}
        python programs/Hierarchical_Impression-CLIP/expt6-2/clustering/tag_feature_similarity.py --dataset ${dataset} --tag_preprocess ${tag_preprocess}
    done
done

# bash programs/Hierarchical_Impression-CLIP/expt6-2/clustering/tag_feature.sh