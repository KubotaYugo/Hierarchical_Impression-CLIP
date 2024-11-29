#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for tag_preprocess in 'normal' 'average_single_tag' 'average_upto_10'; do
    for dataset in 'test'; do
        for random_seed in 1 2 3; do        
            # python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/save_features.py --tag_preprocess ${tag_preprocess} --dataset ${dataset} --random_seed ${random_seed}
            echo "----------------------------------------------------------------------------------------------------"
            echo "tag_preprocess=$tag_preprocess, dataset=$dataset, random_seed=$random_seed"
            python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/quantitative_evaluation.py --tag_preprocess ${tag_preprocess} --dataset ${dataset} --random_seed ${random_seed}
        done
    done
done

# bash programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/evaluation.sh