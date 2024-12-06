#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ベストなもので比較
# 種類別のタグの頻度，全体に対する割合，個数の分布, どのクラスタに属するかを保存

python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/PCA.py --loss_type average   --weights 1.0 0.0 0.0 --dataset train --random_seed 1
python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/PCA.py --loss_type average   --weights 1.0 1.0 1.0 --dataset train --random_seed 3
python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/PCA.py --loss_type iterative --weights 1.0 1.0 1.0 --dataset train --random_seed 5
python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/PCA.py --loss_type label_and --weights 1.0 1.0 1.0 --dataset train --random_seed 5
python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/PCA.py --loss_type average   --weights 1.0 0.0 0.0 --dataset val --random_seed 1
python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/PCA.py --loss_type average   --weights 1.0 1.0 1.0 --dataset val --random_seed 3
python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/PCA.py --loss_type iterative --weights 1.0 1.0 1.0 --dataset val --random_seed 5
python programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/PCA.py --loss_type label_and --weights 1.0 1.0 1.0 --dataset val --random_seed 5

# cd /media/user/0880c7a2-9e74-4110-a983-e8fd1fe49322/hierarchical-CLIP
# conda activate myenv
# bash programs/Hierarchical_Impression-CLIP/expt6-2/evaluation/evaluation4.sh