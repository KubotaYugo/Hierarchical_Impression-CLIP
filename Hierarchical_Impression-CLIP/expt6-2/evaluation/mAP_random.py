'''
類似度行列が乱数の場合の以下のmAP(mean Average Precision)を計算する

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
IMG_CLUSTER_PATH = params.img_cluster_path
TAG_CLUSTER_PATH = params.tag_cluster_path

# クラスタIDの読み込み
img_cluster = np.load(IMG_CLUSTER_PATH)["arr_0"].astype(np.int16())
tag_cluster = np.load(TAG_CLUSTER_PATH)["arr_0"].astype(np.int16())

K = 1000
mAP_pair_tag2img = []
mAP_pair_img2tag = []
mAP_pair_mean = []
mAP_cluster_tag2img = []
mAP_cluster_img2tag = []
mAP_cluster_mean = []
mAP_both_tag2img = []
mAP_both_img2tag = []
mAP_both_mean = []
for i in range(K):
    print(i)
    # Retrieval Rankの計算 (類似度行列の各要素を乱数とする)
    similarity_matrix = (torch.rand(img_cluster.shape[0], img_cluster.shape[0]) * 2) - 1
    RR_tag2img = eval_utils.retrieval_rank_matrix(similarity_matrix, 'tag2img')
    RR_img2tag = eval_utils.retrieval_rank_matrix(similarity_matrix, 'img2tag')

    # 評価指標の計算
    pair_label = np.arange(RR_tag2img.shape[0])                     # ペアのラベル
    mAP_pair_tag2img.append(calc_mAP(RR_tag2img, pair_label))             # ペアの画像のみを正例としたときのmAP(印象->画像)
    mAP_pair_img2tag.append(calc_mAP(RR_img2tag, pair_label))             # ペアの印象のみを正例としたときのmAP(画像->印象)
    mAP_pair_mean.append((mAP_pair_tag2img[i]+mAP_pair_img2tag[i])/2)           # 上2つの平均

    mAP_cluster_tag2img.append(calc_mAP(RR_tag2img, img_cluster))         # 同じクラスタの画像を正例としたときのmAP(印象->画像)
    mAP_cluster_img2tag.append(calc_mAP(RR_img2tag, tag_cluster))         # 同じクラスタの印象を正例としたときのmAP(画像->印象)
    mAP_cluster_mean.append((mAP_cluster_tag2img[i]+mAP_cluster_img2tag[i])/2)  # 上2つの平均

    img_tag_cluster = img_cluster*10+tag_cluster                    # 画像クラスタ, 印象クラスタともに同じものを1つのクラスタとしたラベル
    mAP_both_tag2img.append(calc_mAP(RR_tag2img, img_tag_cluster))        # 画像も印象も同じクラスタのフォントを正例としたときのmAP(印象->画像)
    mAP_both_img2tag.append(calc_mAP(RR_img2tag, img_tag_cluster))        # 画像も印象も同じクラスタのフォントを正例としたときのmAP(画像->印象)
    mAP_both_mean.append((mAP_both_tag2img[i]+mAP_both_img2tag[i])/2)           # 上2つの平均

# numpyに変換
mAP_pair_tag2img = np.asarray(mAP_pair_tag2img)
mAP_pair_img2tag = np.asarray(mAP_pair_img2tag)
mAP_pair_mean = np.asarray(mAP_pair_mean)
mAP_cluster_tag2img = np.asarray(mAP_cluster_tag2img)
mAP_cluster_img2tag = np.asarray(mAP_cluster_img2tag)
mAP_cluster_mean = np.asarray(mAP_cluster_mean)
mAP_both_tag2img = np.asarray(mAP_both_tag2img)
mAP_both_img2tag = np.asarray(mAP_both_img2tag)
mAP_both_mean = np.asarray(mAP_both_mean)

# 結果の表示
print('-----ペアの画像/印象を正解としたとき-----')
print(f'mAP_tag2img: {mAP_pair_tag2img.mean():.6f}')
print(f'mAP_img2tag: {mAP_pair_img2tag.mean():.6f}')
print(f'mAP_mean:    {mAP_pair_mean.mean():.6f}')

print('-----同じクラスタの画像/印象を正解としたとき-----')
print(f'mAP_tag2img: {mAP_cluster_tag2img.mean():.6f}')
print(f'mAP_img2tag: {mAP_cluster_img2tag.mean():.6f}')
print(f'mAP_mean:    {mAP_cluster_mean.mean():.6f}')

print('-----画像も印象も同じクラスタのフォントを正解としたとき-----')
print(f'mAP_tag2img: {mAP_both_tag2img.mean():.6f}')
print(f'mAP_img2tag: {mAP_both_img2tag.mean():.6f}')
print(f'mAP_mean:    {mAP_both_mean.mean():.6f}')