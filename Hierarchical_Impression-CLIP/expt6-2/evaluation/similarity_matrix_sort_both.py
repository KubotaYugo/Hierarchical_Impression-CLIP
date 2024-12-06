'''
各行/各列を画像のクラスタおよび印象のクラスタでソートして類似度行列を作成
'''

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


def save_similarity_matrix(feature1=None, feature2=None, filename=None, 
                           ticks=None, xlabel='Impression feature', ylabel='Image feature'):
    similarity_matrix = torch.matmul(feature1, feature2.T).to("cpu").detach().numpy()
    sns.heatmap(similarity_matrix, cmap='viridis', square=True, 
                cbar=True, vmin=np.min(similarity_matrix), vmax=np.max(similarity_matrix))
    plt.gca().xaxis.set_ticks_position('top') 
    plt.gca().xaxis.set_label_position('top') 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks([])
    plt.yticks([])
    if ticks!=None:
        plt.xticks(ticks=ticks)
        plt.yticks(ticks=ticks) 
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


# define constant
params = utils.get_parameters()
DATASET = params.dataset
BASE_DIR = params.base_dir
IMG_CLUSTER_PATH = params.img_cluster_path
TAG_CLUSTER_PATH = params.tag_cluster_path
NUM_IMG_CLUSTERS = params.num_img_clusters
NUM_TAG_CLUSTERS = params.num_tag_clusters
EMBEDDED_IMG_FEATURE_PATH = params.embedded_img_feature_path
EMBEDDED_TAG_FEATURE_PATH = params.embedded_tag_feature_path

# ディレクトリの作成
SAVE_DIR = f'{BASE_DIR}/similarity_matrix_sort_both/{DATASET}'
os.makedirs(SAVE_DIR, exist_ok=True)

# クラスターIDの読み込み & 両方のクラスタのラベルの作成
img_cluster = np.load(IMG_CLUSTER_PATH)["arr_0"].astype('U')
tag_cluster = np.load(TAG_CLUSTER_PATH)["arr_0"].astype('U')
img_tag_cluster = [str(img_cluster[i])+str(tag_cluster[i]) for i in range(img_cluster.shape[0])]
tag_img_cluster = [str(img_cluster[i])+str(tag_cluster[i]) for i in range(tag_cluster.shape[0])]
img_tag_cluster = np.asarray(img_tag_cluster)
tag_img_cluster = np.asarray(tag_img_cluster)

# 類似度行列の作成
embedded_img_feature = torch.load(EMBEDDED_IMG_FEATURE_PATH)
embedded_tag_feature = torch.load(EMBEDDED_TAG_FEATURE_PATH)
# save_similarity_matrix(embedded_img_feature, embedded_tag_feature, f'{SAVE_DIR}/index.png')

# 画像->印象のクラスタでソート
number_of_instance = np.asarray([np.sum(img_tag_cluster==str(i)+str(j)) for i in range(NUM_IMG_CLUSTERS) for j in range(NUM_TAG_CLUSTERS)])
number_of_instance_cumulative = [np.sum(number_of_instance[:i]) for i in range(len(number_of_instance)+1)]
sorted_indices = np.argsort(img_tag_cluster)
embedded_img_feature_sorted = embedded_img_feature[sorted_indices]
embedded_tag_feature_sorted = embedded_tag_feature[sorted_indices]
save_similarity_matrix(embedded_img_feature_sorted, embedded_tag_feature_sorted, 
                       f'{SAVE_DIR}/img_tag_cluster.png', number_of_instance_cumulative)
save_similarity_matrix(embedded_img_feature_sorted, embedded_img_feature_sorted, 
                       f'{SAVE_DIR}/img2img_img_tag_cluster.png', number_of_instance_cumulative,
                       xlabel='Image feature', ylabel='Image feature')
save_similarity_matrix(embedded_tag_feature_sorted, embedded_tag_feature_sorted, 
                       f'{SAVE_DIR}/tag2tag_img_tag_cluster.png', number_of_instance_cumulative,
                       xlabel='Impression feature', ylabel='Impression feature')

# 印象のクラスタでソート
number_of_instance = np.asarray([np.sum(tag_img_cluster==str(i)+str(j)) for i in range(NUM_IMG_CLUSTERS) for j in range(NUM_TAG_CLUSTERS)])
number_of_instance_cumulative = [np.sum(number_of_instance[:i]) for i in range(len(number_of_instance)+1)]
sorted_indices = np.argsort(tag_img_cluster)
embedded_img_feature_sorted = embedded_img_feature[sorted_indices]
embedded_tag_feature_sorted = embedded_tag_feature[sorted_indices]
save_similarity_matrix(embedded_img_feature_sorted, embedded_tag_feature_sorted, 
                       f'{SAVE_DIR}/tag_img_cluster.png', number_of_instance_cumulative)
save_similarity_matrix(embedded_img_feature_sorted, embedded_img_feature_sorted, 
                       f'{SAVE_DIR}/img2img_tag_img_cluster.png', number_of_instance_cumulative,
                       xlabel='Image feature', ylabel='Image feature')
save_similarity_matrix(embedded_tag_feature_sorted, embedded_tag_feature_sorted, 
                       f'{SAVE_DIR}/tag2tag_tag_img_cluster.png', number_of_instance_cumulative,
                       xlabel='Impression feature', ylabel='Impression feature')
