#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# lib/utils.pyのparamsを以下のようにすること
# 'tag_preprocess': 'average_single_tag'
# 'dataset': 'val'
# 'weights': [1.0, 0.0, 0.0]

# 実行するプログラム
program="similarity_hist_hierarchical"

for initial_temperature in 0.02 0.05 0.07 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    for random_seed in 1 2 3; do
        echo "----------------------------------------------------------------------------------------------------"
        echo "initial_temperature=$initial_temperature, random_seed=$random_seed"
        python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/$program.py --initial_temperature ${initial_temperature} --random_seed ${random_seed}
    done
done

# for initial_temperature in 0.02 0.05 0.07 0.1 0.15 0.2 0.3 0.4; do
#     for random_seed in 4 5; do
#         echo "----------------------------------------------------------------------------------------------------"
#         echo "initial_temperature=$initial_temperature, random_seed=$random_seed"
#         python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/$program.py --initial_temperature ${initial_temperature} --random_seed ${random_seed}
#     done
# done


# bash programs/Hierarchical_Impression-CLIP/expt6-2/temperature_evaluation/temperature_evaluation.sh