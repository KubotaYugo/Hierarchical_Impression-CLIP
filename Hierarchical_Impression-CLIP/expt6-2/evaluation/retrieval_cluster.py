'''
ある画像/印象のペアの印象/画像と同じクラスタの印象/画像をポジティブ，その他をネガティブとして，
recall@k, precision@k, average precisionを求める
それぞれ，クラスタ毎にプロット
'''
import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils
from lib import eval_utils


def calc_metrics(retrieval_rank, clusterID):
    # positive_frag, cumulative_positive
    clusterID = torch.from_numpy(clusterID).to('cuda')
    clusterID_sorted = clusterID[retrieval_rank]
    positive_frag = (clusterID.unsqueeze(1)==clusterID_sorted).int()
    cumulative_positive = torch.cumsum(positive_frag, dim=1) 

    # recall
    relevant_items_per_query = positive_frag.sum(dim=1, keepdim=True)
    recall = cumulative_positive / relevant_items_per_query

    # precision
    rank_indices = torch.arange(1, positive_frag.size(1)+1).to('cuda')
    precision = cumulative_positive/rank_indices                  
    precision[positive_frag==0] = 0

    # average precision
    average_precision = (precision*positive_frag).sum(dim=1)/positive_frag.sum(dim=1)

    recall = recall.to('cpu').detach().numpy()
    precision = precision.to('cpu').detach().numpy()
    average_precision = average_precision.to('cpu').detach().numpy()

    metrics = {'recall': recall,
               'precision': precision,
               'average_precision': average_precision}

    return metrics

def plot_recall(recall, clusterID, save_dir):
    # recall@kを描画
    os.makedirs(save_dir, exist_ok=True)
    for i in range(10):
        plt.plot(recall[clusterID==i].T, c=plt.cm.tab10(0), linewidth=0.08)
        plt.savefig(f'{save_dir}/cluster{i+1}.png', bbox_inches='tight', dpi=300)
        plt.close()

def plot_precision(precision, clusterID, save_dir):
    # precision@kを描画
    os.makedirs(save_dir, exist_ok=True)
    for i in range(10):
        for series in precision[clusterID==i]:
            nonzero_indices = series!=0
            x = np.where(nonzero_indices)[0]+1
            y = series[nonzero_indices]
            
            # 始点がない場合，始点を追加
            if x[0]!=1:
                x = np.insert(x, 0, 0)
                y = np.insert(y, 0, 0)
            # 終点がない場合，終点を追加
            if x[-1] != len(clusterID):
                x = np.append(x, len(clusterID))
                y = np.append(y, (clusterID==i).sum()/len(clusterID))
            
            plt.plot(x, y, c=plt.cm.tab10(0), linewidth=0.08)

        plt.savefig(f'{save_dir}/cluster{i+1}.png', bbox_inches='tight', dpi=300)
        plt.close()

def plot_average_precision(average_precision, clusterID, save_dir):
    # average_precisionの描画
    for i in range(10):
        average_precision_sorted = np.sort(average_precision[clusterID==i])[::-1]
        plt.plot(average_precision_sorted, c=plt.cm.tab10(i), label=f'cluster{i+1}')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f'{save_dir}/average_precision.png', bbox_inches='tight', dpi=300)
    plt.close()


# define constant
params = utils.get_parameters()
DATASET = params.dataset
BASE_DIR = params.base_dir
IMG_CLUSTER_PATH = params.img_cluster_path
TAG_CLUSTER_PATH = params.tag_cluster_path
EMBEDDED_IMG_FEATURE_PATH = params.embedded_img_feature_path
EMBEDDED_TAG_FEATURE_PATH = params.embedded_tag_feature_path

# 特徴量の読み込み
embedded_img_feature = torch.load(EMBEDDED_IMG_FEATURE_PATH)
embedded_tag_feature = torch.load(EMBEDDED_TAG_FEATURE_PATH)

# クラスタIDの読み込み
img_cluster = np.load(IMG_CLUSTER_PATH)["arr_0"].astype(np.int16())
tag_cluster = np.load(TAG_CLUSTER_PATH)["arr_0"].astype(np.int16())

# Retrieval Rankの計算
similarity_matrix = torch.matmul(embedded_img_feature, embedded_tag_feature.T)
RR_matrix_tag2img = eval_utils.retrieval_rank_matrix(similarity_matrix, 'tag2img')
RR_matrix_img2tag = eval_utils.retrieval_rank_matrix(similarity_matrix, 'img2tag')

# recall, precision, average precisionの計算
metrics_tag2img = calc_metrics(RR_matrix_tag2img, img_cluster)
metrics_img2tag = calc_metrics(RR_matrix_img2tag, tag_cluster)

# tag2imgの評価指標をプロットして保存
SAVE_DIR = f'{BASE_DIR}/retrieval_cluster/tag2img'
plot_recall(metrics_tag2img['recall'], img_cluster, f'{SAVE_DIR}/recall/{DATASET}')
plot_precision(metrics_tag2img['precision'], img_cluster, f'{SAVE_DIR}/precision/{DATASET}')
plot_average_precision(metrics_tag2img['average_precision'], img_cluster, SAVE_DIR)

# img2tagの評価指標をプロットして保存
SAVE_DIR = f'{BASE_DIR}/retrieval_cluster/img2tag'
plot_recall(metrics_img2tag['recall'], tag_cluster, f'{SAVE_DIR}/recall/{DATASET}')
plot_precision(metrics_img2tag['precision'], tag_cluster, f'{SAVE_DIR}/precision/{DATASET}')
plot_average_precision(metrics_img2tag['average_precision'], tag_cluster, SAVE_DIR)