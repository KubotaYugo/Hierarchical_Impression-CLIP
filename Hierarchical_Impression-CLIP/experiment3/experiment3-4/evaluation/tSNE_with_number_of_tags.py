'''
印象タグの個数で色分けして印象特徴をプロット
'''
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import torch
import numpy as np
import matplotlib.pyplot as plt
from lib import utils
from lib import eval_utils



# define constant
EXP = utils.EXP
IMG_CLUSTER_PATH = utils.IMG_CLUSTER_PATH
TAG_CLUSTER_PATH = utils.TAG_CLUSTER_PATH
BATCH_SIZE = utils.BATCH_SIZE
BASE_DIR = utils.BASE_DIR

for DATASET in ['train', 'val', 'test']:
    # 保存用フォルダの準備
    SAVE_DIR = f"{BASE_DIR}/tSNE_with_number_of_tags"
    os.makedirs(f"{SAVE_DIR}", exist_ok=True)
    
    # 特徴量の読み込み
    load_dir = f'{BASE_DIR}/features/{DATASET}'
    embedded_img_features = torch.load(f'{load_dir}/embedded_img_features.pth')
    embedded_tag_features = torch.load(f'{load_dir}/embedded_tag_features.pth')

    # Retrieval Rankの計算
    similarity_matrix = torch.matmul(embedded_img_features, embedded_tag_features.T)
    RR_img2tag = eval_utils.retrieval_rank(similarity_matrix, "img2tag")
    RR_tag2img = eval_utils.retrieval_rank(similarity_matrix, "tag2img")

    # tSNE特徴量の読み込み
    tSNE_feature_filename = f'{BASE_DIR}/tSNE/{DATASET}/tSNE_feature.npz'
    tSNE_embedding = np.load(tSNE_feature_filename)['arr_0']

    # タグの個数を取得
    _, tag_paths = utils.load_dataset_paths(DATASET)
    number_of_tags = [len(utils.get_font_tags(tag_path)) for tag_path in tag_paths]
    number_of_tags = np.asarray(number_of_tags)

    # タグの個数で色分けして，印象特徴をプロット
    x = tSNE_embedding[:, 0]
    y = tSNE_embedding[:, 1]
    tag_x = x[len(embedded_img_features):]
    tag_y = y[len(embedded_img_features):]
    x_min = min(x)-5
    x_max = max(x)+5
    y_min = min(y)-5
    y_max = max(y)+5

    fig, ax = plt.subplots(figsize=(6.4*1.5, 4.8*1.5))
    for i in range(1, 10+1):
        plt.scatter(tag_x[number_of_tags==i], tag_y[number_of_tags==i], alpha=0.8, edgecolors='w', linewidths=0.1, s=5, label=f'{i}')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.savefig(f"{SAVE_DIR}/{DATASET}.png", bbox_inches='tight', dpi=500)
    plt.close()