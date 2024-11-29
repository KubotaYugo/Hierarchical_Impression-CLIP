import torch
import numpy as np

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils
from lib import eval_utils


# define constant
params = utils.get_parameters()
DATASET = params.dataset
BASE_DIR = params.base_dir
EMBEDDED_IMG_FEATURE_PATH = params.embedded_img_feature_path
EMBEDDED_TAG_FEATURE_PATH = params.embedded_tag_feature_path
EMBEDDED_SINGLE_TAG_FEATURE_PATH = params.embedded_single_tag_feature_path

# 特徴量の読み込み
embedded_img_feature = torch.load(EMBEDDED_IMG_FEATURE_PATH)
embedded_tag_feature = torch.load(EMBEDDED_TAG_FEATURE_PATH)
embedded_single_tag_feature = torch.load(EMBEDDED_SINGLE_TAG_FEATURE_PATH)

# Retrieval Rankの計算
similarity_matrix = torch.matmul(embedded_img_feature, embedded_tag_feature.T)
RR_img2tag = eval_utils.retrieval_rank(similarity_matrix, "img2tag")
RR_tag2img = eval_utils.retrieval_rank(similarity_matrix, "tag2img")

# mean Average Precisionの計算
img_paths, tag_paths = utils.load_dataset_paths(DATASET)
tag_list = list(utils.get_tag_list())
AP_tag2img = eval_utils.AP_tag2img(embedded_img_feature, embedded_single_tag_feature, tag_list, tag_paths)
AP_img2tag = eval_utils.AP_img2tag(embedded_img_feature, embedded_single_tag_feature, tag_list, tag_paths)

print(f'-----{DATASET}-----')
print(f"meanARR:     {(np.mean(RR_tag2img)+np.mean(RR_img2tag))/2:.2f}")
print(f"ARR_tag2img: {np.mean(RR_tag2img):.2f}")
print(f"ARR_img2tag: {np.mean(RR_img2tag):.2f}")
print(f"AP_tag2img:  {np.mean(AP_tag2img):.4f}")
print(f"AP_img2tag:  {np.mean(AP_img2tag):.4f}")