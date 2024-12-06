#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

loss_type="average"
weights="1.0 0.0 0.0"
for dataset in 'val'; do
    for random_seed in 1 2 3 4 5; do    
        # echo "----------------------------------------------------------------------------------------------------"
        # echo "loss_type=$loss_type, --weights $weights, dataset=$dataset, random_seed=$random_seed"
        python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/mAP.py --loss_type ${loss_type} --weights ${weights} --dataset ${dataset} --random_seed ${random_seed}
    done
done


weights="1.0 1.0 1.0"
for loss_type in 'average' 'iterative' 'label_and'; do
    for dataset in 'val'; do
        for random_seed in 1 2 3 4 5; do
            # echo "----------------------------------------------------------------------------------------------------"
            # echo "loss_type=$loss_type, --weights $weights, dataset=$dataset, random_seed=$random_seed"
            python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/mAP.py --loss_type ${loss_type} --weights ${weights} --dataset ${dataset} --random_seed ${random_seed}
        done
    done
done

# python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/retrieve.py --loss_type average   --weights 1.0 0.0 0.0 --dataset val --random_seed 1
# python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/retrieve.py --loss_type average   --weights 1.0 1.0 1.0 --dataset val --random_seed 3
# python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/retrieve.py --loss_type iterative --weights 1.0 1.0 1.0 --dataset val --random_seed 5
# python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/retrieve.py --loss_type label_and --weights 1.0 1.0 1.0 --dataset val --random_seed 5

# cd /media/user/0880c7a2-9e74-4110-a983-e8fd1fe49322/hierarchical-CLIP
# conda activate myenv
# bash programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/evaluation3.sh

