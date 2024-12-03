#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# for loss_type in 'average' 'iterative' 'label_and'; do
#     for dataset in 'train' 'val'; do
#         for random_seed in 1 2 3 4 5; do        
#             # python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/save_features.py --tag_preprocess ${tag_preprocess} --dataset ${dataset} --random_seed ${random_seed}
#             echo "----------------------------------------------------------------------------------------------------"
#             echo "loss_type=$loss_type, dataset=$dataset, random_seed=$random_seed"
#             python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/save_features.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed}
#             # python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/quantitative_evaluation.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed}
#             python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/PCA.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed}
#             python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/tSNE.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed}
#             python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/similarity_hist.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed}
#             python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/similarity_hist_hierarchical.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed}
#             python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/similarity_matrix.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed}
#             python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/rank_matrix.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed}
#             python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/retrieval_rank_hist.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed}
#             python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/retrieval_cluster.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed}
#             python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/PCA_colored_by_pair.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed}
#             python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/retrieve.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed}
#         done
#     done
# done

loss_type="average"
weights="1.0 0.0 0.0"

for dataset in 'train' 'val'; do
    for random_seed in 1 2 3 4 5; do    
        echo "----------------------------------------------------------------------------------------------------"
        echo "loss_type=$loss_type, dataset=$dataset, weights=$weights, random_seed=$random_seed"
        python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/save_features.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed} --weights ${weights}
        # python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/quantitative_evaluation.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed} --weights ${weights}
        python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/PCA.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed} --weights ${weights}
        python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/tSNE.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed} --weights ${weights}
        python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/similarity_hist.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed} --weights ${weights}
        python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/similarity_hist_hierarchical.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed} --weights ${weights}
        python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/similarity_matrix.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed} --weights ${weights}
        python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/rank_matrix.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed} --weights ${weights}
        python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/retrieval_rank_hist.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed} --weights ${weights}
        python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/retrieval_cluster.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed} --weights ${weights}
        python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/PCA_colored_by_pair.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed} --weights ${weights}
        python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/retrieve.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed} --weights ${weights}
    done
done


# for loss_type in 'average' 'iterative' 'label_and'; do
#     for dataset in 'train' 'val'; do
#         for random_seed in 1 2 3 4 5; do        
#             # python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/save_features.py --tag_preprocess ${tag_preprocess} --dataset ${dataset} --random_seed ${random_seed}
#             echo "----------------------------------------------------------------------------------------------------"
#             echo "loss_type=$loss_type, dataset=$dataset, random_seed=$random_seed"
#             python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/retrieve.py --loss_type ${loss_type} --dataset ${dataset} --random_seed ${random_seed}
#         done
#     done
# done


# cd /media/user/0880c7a2-9e74-4110-a983-e8fd1fe49322/hierarchical-CLIP
# conda activate myenv
# bash programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/evaluation.sh