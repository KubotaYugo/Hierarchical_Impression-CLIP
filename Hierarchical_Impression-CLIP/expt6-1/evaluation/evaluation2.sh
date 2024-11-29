#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for dataset in 'train' 'val' 'test'; do
    for loss_type in 'average' 'iterative' 'label_and'; do
        # python programs/Hierarchical_Impression-CLIP/expt6-1/evaluation/save_features.py --loss_type ${loss_type} --dataset ${dataset}
        # python programs/Hierarchical_Impression-CLIP/expt6-1/evaluation/similarity_matrix.py --loss_type ${loss_type} --dataset ${dataset}
        # python programs/Hierarchical_Impression-CLIP/expt6-1/evaluation/tSNE.py --loss_type ${loss_type} --dataset ${dataset}
        # python programs/Hierarchical_Impression-CLIP/expt6-1/evaluation/PCA.py --loss_type ${loss_type} --dataset ${dataset}
        python programs/Hierarchical_Impression-CLIP/expt6-1/evaluation/similarity_hist_hierarchical.py --loss_type ${loss_type} --dataset ${dataset}
    done
    # python programs/Hierarchical_Impression-CLIP/expt6-1/evaluation/save_features.py --weights 1.0 0.0 0.0 --loss_type average --ce_bce CE --dataset ${dataset}
    # python programs/Hierarchical_Impression-CLIP/expt6-1/evaluation/similarity_matrix.py --weights 1.0 0.0 0.0 --loss_type average --ce_bce CE --dataset ${dataset}
    # python programs/Hierarchical_Impression-CLIP/expt6-1/evaluation/tSNE.py --weights 1.0 0.0 0.0 --loss_type average --ce_bce CE --dataset ${dataset}
    # python programs/Hierarchical_Impression-CLIP/expt6-1/evaluation/PCA.py --weights 1.0 0.0 0.0 --loss_type average --ce_bce CE --dataset ${dataset}
    python programs/Hierarchical_Impression-CLIP/expt6-1/evaluation/similarity_hist_hierarchical.py --weights 1.0 0.0 0.0 --loss_type average --ce_bce CE --dataset ${dataset}
done

# bash programs/Hierarchical_Impression-CLIP/expt6-1/evaluation/evaluation2.sh