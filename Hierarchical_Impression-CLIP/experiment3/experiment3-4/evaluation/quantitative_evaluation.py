import torch
import numpy as np

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils
from lib import eval_utils


# define constant
BATCH_SIZE = utils.BATCH_SIZE
MODEL_PATH = utils.MODEL_PATH
BASE_DIR = utils.BASE_DIR

for DATASET in ['train', 'val', 'test']:
    # 特徴量の読み込み
    load_dir = f'{BASE_DIR}/features/{DATASET}'
    embedded_img_features = torch.load(f'{load_dir}/embedded_img_features.pth')
    embedded_tag_features = torch.load(f'{load_dir}/embedded_tag_features.pth')
    embedded_single_tag_features = torch.load(f'{load_dir}/embedded_single_tag_features.pth')

    # Average Retrieval Rankの計算
    similarity_matrix = torch.matmul(embedded_img_features, embedded_tag_features.T)
    RR_img2tag = eval_utils.retrieval_rank(similarity_matrix, "img2tag")
    RR_tag2img = eval_utils.retrieval_rank(similarity_matrix, "tag2img")

    # mean Average Precisionの計算
    img_paths, tag_paths = utils.load_dataset_paths(DATASET)
    tag_list = list(eval_utils.get_tag_list().values())
    AP_tag2img = eval_utils.AP_tag2img(embedded_img_features, embedded_single_tag_features, tag_list, tag_paths)
    AP_img2tag = eval_utils.AP_img2tag(embedded_img_features, embedded_single_tag_features, tag_list, tag_paths)

    print(f'-----{DATASET}-----')
    print(f"meanARR:     {(np.mean(RR_tag2img)+np.mean(RR_img2tag))/2:.2f}")
    print(f"ARR_tag2img: {np.mean(RR_tag2img):.2f}")
    print(f"ARR_img2tag: {np.mean(RR_img2tag):.2f}")
    print(f"AP_tag2img:  {np.mean(AP_tag2img):.4f}")
    print(f"AP_img2tag:  {np.mean(AP_img2tag):.4f}")