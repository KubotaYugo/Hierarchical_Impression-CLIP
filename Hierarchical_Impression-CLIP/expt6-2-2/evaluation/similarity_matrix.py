import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils
import seaborn as sns


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
    plt.close()


# define constant
params = utils.get_parameters()
DATASET = params.dataset
BASE_DIR = params.base_dir
IMG_CLUSTER_PATH = params.img_cluster_path
TAG_CLUSTER_PATH = params.tag_cluster_path
EMBEDDED_IMG_FEATURE_PATH = params.embedded_img_feature_path
EMBEDDED_TAG_FEATURE_PATH = params.embedded_tag_feature_path

# ディレクトリの作成
SAVE_DIR = f'{BASE_DIR}/similarity_matrix/{DATASET}'
os.makedirs(SAVE_DIR, exist_ok=True)

# 類似度行列の作成
embedded_img_feature = torch.load(EMBEDDED_IMG_FEATURE_PATH)
embedded_tag_feature = torch.load(EMBEDDED_TAG_FEATURE_PATH)
save_similarity_matrix(embedded_img_feature, embedded_tag_feature, f'{SAVE_DIR}/index.png')

# 画像のクラスタでソート
img_cluster = np.load(IMG_CLUSTER_PATH)["arr_0"].astype(np.int16())
number_of_instance = np.asarray([np.sum(img_cluster==i) for i in range(10)])
number_of_instance_cumulative = [np.sum(number_of_instance[:i+1]) for i in range(9)]
sorted_indices = np.argsort(img_cluster)
embedded_img_feature_sorted = embedded_img_feature[sorted_indices]
embedded_tag_feature_sorted = embedded_tag_feature[sorted_indices]
save_similarity_matrix(embedded_img_feature_sorted, embedded_tag_feature_sorted, 
                       f'{SAVE_DIR}/img_cluster.png', number_of_instance_cumulative)
save_similarity_matrix(embedded_img_feature_sorted, embedded_img_feature_sorted, 
                       f'{SAVE_DIR}/img2img_img_cluster.png', number_of_instance_cumulative,
                       xlabel='Image feature', ylabel='Image feature')
save_similarity_matrix(embedded_tag_feature_sorted, embedded_tag_feature_sorted, 
                       f'{SAVE_DIR}/tag2tag_img_cluster.png', number_of_instance_cumulative,
                       xlabel='Impression feature', ylabel='Impression feature')

# 印象のクラスタでソート
tag_cluster = np.load(TAG_CLUSTER_PATH)["arr_0"].astype(np.int16())
number_of_instance = np.asarray([np.sum(tag_cluster==i) for i in range(10)])
number_of_instance_cumulative = [np.sum(number_of_instance[:i+1]) for i in range(9)]
sorted_indices = np.argsort(tag_cluster)
embedded_img_feature_sorted = embedded_img_feature[sorted_indices]
embedded_tag_feature_sorted = embedded_tag_feature[sorted_indices]
save_similarity_matrix(embedded_img_feature_sorted, embedded_tag_feature_sorted, 
                       f'{SAVE_DIR}/tag_cluster.png', number_of_instance_cumulative)
save_similarity_matrix(embedded_img_feature_sorted, embedded_img_feature_sorted, 
                       f'{SAVE_DIR}/img2img_tag_cluster.png', number_of_instance_cumulative,
                       xlabel='Image feature', ylabel='Image feature')
save_similarity_matrix(embedded_tag_feature_sorted, embedded_tag_feature_sorted, 
                       f'{SAVE_DIR}/tag2tag_tag_cluster.png', number_of_instance_cumulative,
                       xlabel='Impression feature', ylabel='Impression feature')