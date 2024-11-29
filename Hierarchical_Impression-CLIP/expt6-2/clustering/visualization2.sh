#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for dataset in 'train' 'val' 'test'; do
    # python programs/Hierarchical_Impression-CLIP/expt6-1/clustering/PCA_img_plot_as_img.py --dataset ${dataset}
    # python programs/Hierarchical_Impression-CLIP/expt6-1/clustering/tSNE_img_plot_as_img.py --dataset ${dataset}
    for tag_preprocess in 'normal' 'average_single_tag' 'average_upto_10'; do
        python programs/Hierarchical_Impression-CLIP/expt6-1/clustering/PCA_tag_number_of_tags.py --dataset ${dataset} --tag_preprocess ${tag_preprocess}
        python programs/Hierarchical_Impression-CLIP/expt6-1/clustering/tSNE_tag_number_of_tags.py --dataset ${dataset} --tag_preprocess ${tag_preprocess}
    done
done

# bash programs/Hierarchical_Impression-CLIP/expt6-1/clustering/visualization2.sh