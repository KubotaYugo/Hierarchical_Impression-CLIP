import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


def save_hist(data, name):
    # ピッタリ1のデータがあったら0.9999999999に (ヒストグラムに反映させるため)
    data = np.asarray(data).copy()
    data[data==1.0] = 0.9999999999
    
    fig, ax = plt.subplots()
    bin_width = BIN_WIDTH
    bins = np.arange(-1, 1+bin_width, bin_width)
    ax.hist(data, bins=bins)
    ax.set_xlabel('Cosine similarity')
    ax.set_ylabel('Frequency')
    plt.savefig(f"{SAVE_DIR}/{DATASET}_{name}.png", bbox_inches='tight', dpi=300)
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
SAVE_DIR = f'{BASE_DIR}/similarity_hist_hierarchical'
os.makedirs(SAVE_DIR, exist_ok=True)

# 特徴量の読み込み & similarityの計算
embedded_img_feature = torch.load(EMBEDDED_IMG_FEATURE_PATH)
embedded_tag_feature = torch.load(EMBEDDED_TAG_FEATURE_PATH)
similarity_matrix = torch.matmul(embedded_img_feature, embedded_tag_feature.T).to("cpu").detach().numpy()

# クラスタの番号を取得
img_hierarchy = np.load(IMG_CLUSTER_PATH)["arr_0"].astype(int)
tag_hierarchy = np.load(TAG_CLUSTER_PATH)["arr_0"].astype(int)

# 画像と印象のそれぞれのモダリティにおけるクラスタが同じかどうか
img_flag = (img_hierarchy[:, None] == img_hierarchy[None, :]).astype(int)
tag_flag = (tag_hierarchy[:, None] == tag_hierarchy[None, :]).astype(int)
total_flag = ((img_flag+tag_flag)==2).astype(int)

# ラベルの作成
# 0: どちらのモダリティでも同じクラスタでない
# 1: 画像が同じクラスタ
# 2: 印象が同じクラスタ
# 3: 両方のモダリティで同じクラスタ
label = np.zeros_like(total_flag)
label[img_flag==1] = 1
label[tag_flag==1] = 2
label[total_flag==1] = 3

# FigureとAxesを生成
fig, ax1 = plt.subplots()
BIN_WIDTH = 0.025

# 負例の類似度を抽出
diag_mask = np.eye(similarity_matrix.shape[0], dtype=bool)
non_diag_mask = ~diag_mask

# 個別のヒストグラムをプロット
save_hist(np.diag(similarity_matrix), 'Positive_pair')
save_hist(similarity_matrix[(non_diag_mask==1)&(label==0)], 'Same_cluster_in_both_modalities')
save_hist(similarity_matrix[(non_diag_mask==1)&(label==1)], 'Same_cluster_in_image_modality')
save_hist(similarity_matrix[(non_diag_mask==1)&(label==2)], 'Same_cluster_in_impression_modality')
save_hist(similarity_matrix[(non_diag_mask==1)&(label==3)], 'Different_cluster_in_both_modalities')

# 全部を1つの画像にまとめてプロット
fig, axs = plt.subplots(5, 1, figsize=(6, 9), constrained_layout=True)
bin_width = BIN_WIDTH
bins = np.arange(-1, 1+bin_width, bin_width)

axs[0].hist(np.diag(similarity_matrix), bins=bins)
axs[0].set_xlabel('Positive pair')
axs[0].set_ylabel('Frequency')

axs[1].hist(similarity_matrix[(non_diag_mask==1)&(label==3)], bins=bins)
axs[1].set_xlabel('Same cluster in both modalities')
axs[1].set_ylabel('Frequency')

axs[2].hist(similarity_matrix[(non_diag_mask==1)&(label==1)], bins=bins)
axs[2].set_xlabel('Same cluster in image modality')
axs[2].set_ylabel('Frequency')

axs[3].hist(similarity_matrix[(non_diag_mask==1)&(label==2)], bins=bins)
axs[3].set_xlabel('Same cluster in impression modality')
axs[3].set_ylabel('Frequency')

axs[4].hist(similarity_matrix[(non_diag_mask==1)&(label==0)], bins=bins)
axs[4].set_xlabel('Different cluster in both modalities')
axs[4].set_ylabel('Frequency')

plt.savefig(f"{SAVE_DIR}/{DATASET}.png", bbox_inches='tight', dpi=300)
plt.close()