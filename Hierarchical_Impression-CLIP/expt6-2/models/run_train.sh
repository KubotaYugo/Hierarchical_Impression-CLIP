#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python programs/Hierarchical_Impression-CLIP/expt6-2/models/train2.py

# cd /media/user/0880c7a2-9e74-4110-a983-e8fd1fe49322/hierarchical-CLIP
# conda activate myenv
# bash programs/Hierarchical_Impression-CLIP/expt6-2/models/run_train.sh