'''
以下のmAP(mean Average Precision)を計算する

ペアの画像のみを正例としたときのmAP(印象->画像)
ペアの印象のみを正例としたときのmAP(画像->印象)
上2つの平均

同じクラスタの画像を正例としたときのmAP(印象->画像)
同じクラスタの印象を正例としたときのmAP(画像->印象)
上2つの平均

画像も印象も同じクラスタのフォントを正例としたときのmAP(印象->画像)
画像も印象も同じクラスタのフォントを正例としたときのmAP(画像->印象)
上2つの平均
'''

import torch
import numpy as np

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils
from lib import eval_utils


def calc_mAP(retrieval_rank, clusterID):
    # Convert clusterID to tensor and move to CUDA
    clusterID = torch.from_numpy(clusterID).to('cuda')
    clusterID_sorted = clusterID[retrieval_rank]
    
    # Calculate positive fragments and cumulative positives
    positive_frag = (clusterID.unsqueeze(1) == clusterID_sorted).int()
    cumulative_positive = torch.cumsum(positive_frag, dim=1)

    # Calculate mean average precision (mAP)
    rank_indices = torch.arange(1, positive_frag.size(1) + 1).to('cuda')
    precision = cumulative_positive / rank_indices                  
    precision[positive_frag == 0] = 0
    average_precision = (precision * positive_frag).sum(dim=1) / positive_frag.sum(dim=1)
    mean_average_precision = average_precision.mean().item()

    return mean_average_precision


# define constant
params = utils.get_parameters()
DATASET = params.dataset
BASE_DIR = params.base_dir
EMBEDDED_IMG_FEATURE_PATH = params.embedded_img_feature_path
EMBEDDED_TAG_FEATURE_PATH = params.embedded_tag_feature_path

# 特徴量の読み込み
embedded_img_feature = torch.load(EMBEDDED_IMG_FEATURE_PATH)
embedded_tag_feature = torch.load(EMBEDDED_TAG_FEATURE_PATH)

# Retrieval Rankの計算
similarity_matrix = torch.matmul(embedded_img_feature, embedded_tag_feature.T)
RR_tag2img = eval_utils.retrieval_rank_matrix(similarity_matrix, 'tag2img')
RR_img2tag = eval_utils.retrieval_rank_matrix(similarity_matrix, 'img2tag')

# 評価指標の計算
pair_label = np.arange(RR_tag2img.shape[0])                     # ペアのラベル
mAP_pair_tag2img = calc_mAP(RR_tag2img, pair_label)             # ペアの画像のみを正例としたときのmAP(印象->画像)
mAP_pair_img2tag = calc_mAP(RR_img2tag, pair_label)             # ペアの印象のみを正例としたときのmAP(画像->印象)
mAP_pair_mean = (mAP_pair_tag2img+mAP_pair_img2tag)/2           # 上2つの平均

# 結果の表示
print('-----ペアの画像/印象を正解としたとき-----')
print(f'mAP_tag2img: {mAP_pair_tag2img:.4f}')
print(f'mAP_img2tag: {mAP_pair_img2tag:.4f}')
print(f'mAP_mean:    {mAP_pair_mean:.4f}')