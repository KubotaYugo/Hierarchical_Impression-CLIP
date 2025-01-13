#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for dataset in 'train' 'val' 'test'; do
    python programs/Hierarchical_Impression-CLIP/expt6-2-2/evaluation/save_features.py --dataset ${dataset}
done

# bash programs/Hierarchical_Impression-CLIP/expt6-2-2/evaluation/save_features.sh