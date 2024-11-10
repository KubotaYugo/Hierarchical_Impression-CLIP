import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils
import seaborn as sns


def save_similarity_matrix(embedded_img_features=None, embedded_tag_features=None, filename=None, ticks=None):
    similarity_matrix = torch.matmul(embedded_img_features, embedded_tag_features.T).to("cpu").detach().numpy()
    sns.heatmap(similarity_matrix, cmap='coolwarm', square=True, 
                cbar=True, vmin=np.min(similarity_matrix), vmax=np.max(similarity_matrix))
    plt.xlabel('Impression feature')
    plt.ylabel('Image feature')
    if ticks!=None:
        plt.xticks(ticks=ticks)
        plt.yticks(ticks=ticks) 
    plt.savefig(f'{SAVE_DIR}/{filename}.png', dpi=500, bbox_inches='tight')
    plt.close()


# define constant
EXP = utils.EXP
LR = utils.LR
BATCH_SIZE = utils.BATCH_SIZE    # フォントの数 (画像と印象を合わせた数はこれの2倍)
MODEL_PATH = f'{EXP}/LR={LR}, BS={BATCH_SIZE}/results/model/best.pth.tar'
IMG_CLUSTER_PATH = 'Hierarchical_Impression-CLIP/experiment3/experiment3-4/clustering/train/image_clusters.npz'
TAG_CLUSTER_PATH = 'Hierarchical_Impression-CLIP/experiment3/experiment3-4/clustering/train/impression_clusters.npz'
DATASET = 'test'

SAVE_DIR = f'{EXP}/LR={LR}, BS={BATCH_SIZE}/similarity_matrix'
os.makedirs(SAVE_DIR, exist_ok=True)

for DATASET in ['train', 'val', 'test']:
    # 類似度行列の作成
    load_dir = f'{EXP}/LR={LR}, BS={BATCH_SIZE}/features/{DATASET}'
    embedded_img_features = torch.load(f'{load_dir}/embedded_img_features.pth')
    embedded_tag_features = torch.load(f'{load_dir}/embedded_tag_features.pth')
    save_similarity_matrix(embedded_img_features, embedded_tag_features, DATASET)

    if DATASET=='train':        
        # 画像のクラスタでソート
        img_cluster = np.load(IMG_CLUSTER_PATH)["arr_0"].astype(np.int16())
        number_of_instance = np.asarray([np.sum(img_cluster==i) for i in range(10)])
        number_of_instance_cumulative = [np.sum(number_of_instance[:i+1]) for i in range(9)]
        sorted_indices = np.argsort(img_cluster)
        embedded_img_features_sorted = embedded_img_features[sorted_indices]
        embedded_tag_features_sorted = embedded_tag_features[sorted_indices]
        save_similarity_matrix(embedded_img_features_sorted, embedded_tag_features_sorted, 
                               f'{DATASET}_img_cluster', number_of_instance_cumulative)

        # 印象のクラスタでソート
        tag_cluster = np.load(TAG_CLUSTER_PATH)["arr_0"].astype(np.int16())
        number_of_instance = np.asarray([np.sum(tag_cluster==i) for i in range(10)])
        number_of_instance_cumulative = [np.sum(number_of_instance[:i+1]) for i in range(9)]
        sorted_indices = np.argsort(tag_cluster)
        embedded_img_features_sorted = embedded_img_features[sorted_indices]
        embedded_tag_features_sorted = embedded_tag_features[sorted_indices]
        save_similarity_matrix(embedded_img_features_sorted, embedded_tag_features_sorted, 
                               f'{DATASET}_tag_cluster', number_of_instance_cumulative)