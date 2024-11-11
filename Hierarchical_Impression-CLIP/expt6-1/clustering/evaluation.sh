#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for dataset in 'train' 'val' 'test'; do
    python programs/Hierarchical_Impression-CLIP/expt6-1/clustering/save_cluster_imgs.py --dataset ${dataset}
    for tag_preprocess in 'normal' 'average_single_tag' 'average_upto_10'; do
        # 'single_tag' には未対応
        python programs/Hierarchical_Impression-CLIP/expt6-1/clustering/save_cluster_tags.py --dataset ${dataset} --tag_preprocess ${tag_preprocess}
        python programs/Hierarchical_Impression-CLIP/expt6-1/clustering/tag_freq.py --dataset ${dataset} --tag_preprocess ${tag_preprocess}
        python programs/Hierarchical_Impression-CLIP/expt6-1/clustering/tag_freq_ratio.py --dataset ${dataset} --tag_preprocess ${tag_preprocess}
        python programs/Hierarchical_Impression-CLIP/expt6-1/clustering/number_of_tags_freq.py --dataset ${dataset} --tag_preprocess ${tag_preprocess}
        python programs/Hierarchical_Impression-CLIP/expt6-1/clustering/cluster_matching.py --dataset ${dataset} --tag_preprocess ${tag_preprocess}
    done
done

# bash programs/Hierarchical_Impression-CLIP/expt6-1/clustering/evaluation.sh