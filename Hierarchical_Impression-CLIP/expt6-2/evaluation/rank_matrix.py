import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


def culc_rank(query_feature, key_feature):
    similarity_matrix = torch.matmul(query_feature, key_feature.T).to("cpu").detach().numpy()
    distances = -similarity_matrix
    nearest_idx = np.argsort(distances, axis=1)
    rank = np.zeros_like(nearest_idx)
    rows, cols = np.indices(nearest_idx.shape)
    rank[rows, nearest_idx] = cols
    return rank

def Plot(img_rank_list, tag_rank_list, number_of_font, MODE):
    V_MIN = 0
    V_MAX = 5000
    V_CENTER = 2500
    FONT_SIZE = 16
    LABELS_FONT_SIZE = 11
    img_rank_list[img_rank_list==0] = 1
    tag_rank_list[tag_rank_list==0] = 1
    heatmap, xedges, yedges = np.histogram2d(img_rank_list, tag_rank_list, bins=34, 
                                             range=[[1, number_of_font-1],[1, number_of_font-1]])
    # ax = sns.heatmap(heatmap, cmap='coolwarm', annot=False, fmt=".2f", 
    #                  vmin=V_MIN, vmax=V_MAX, center=V_CENTER, square=True)
    ax = sns.heatmap(heatmap, cmap='coolwarm', annot=False, fmt=".2f", square=True)
    ax.invert_yaxis()
    ticks = [x for x in range(0, len(xedges), 2)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([math.ceil(xedges[i]) for i in range(0, len(xedges), 2)], 
                       rotation=90, fontsize=LABELS_FONT_SIZE)
    ax.set_yticklabels([math.ceil(yedges[i]) for i in range(0, len(xedges), 2)], 
                       rotation=0, fontsize=LABELS_FONT_SIZE)
    ax.figure.axes[-1].tick_params(axis='x', pad=0)
    ax.figure.axes[-1].tick_params(axis='y', pad=0)
    ax.set_xlabel('Image-based rank', fontsize=FONT_SIZE)
    ax.set_ylabel('Impression-based rank', fontsize=FONT_SIZE)
    # cbar = ax.collections[0].colorbar
    # cbar.set_ticks([0, 1000, 2000, 3000, 4000, 5000])
    # cbar.set_ticklabels([' 0', ' 1000', ' 2000', ' 3000', ' 4000', ' ≥5000'])
    plt.savefig(f"{SAVE_DIR}/{MODE}.png", dpi=300, bbox_inches="tight")
    plt.close()



# define constant
params = utils.get_parameters()
DATASET = params.dataset
BASE_DIR = params.base_dir
IMG_FEATURE_PATH = params.img_feature_path
TAG_FEATURE_PATH = params.tag_feature_path
EMBEDDED_IMG_FEATURE_PATH = params.embedded_img_feature_path
EMBEDDED_TAG_FEATURE_PATH = params.embedded_tag_feature_path

# ディレクトリの作成
SAVE_DIR = f'{BASE_DIR}/rank_matrix/{DATASET}'
os.makedirs(SAVE_DIR, exist_ok=True)

# 特徴量の読み込み
img_feature = torch.load(IMG_FEATURE_PATH)
tag_feature = torch.load(TAG_FEATURE_PATH)
embedded_img_feature = torch.load(EMBEDDED_IMG_FEATURE_PATH)
embedded_tag_feature = torch.load(EMBEDDED_TAG_FEATURE_PATH)

# フォント数を計算
number_of_font = len(img_feature)

# 埋め込み前の検索順位
img_rank_before = culc_rank(img_feature, img_feature)
tag_rank_before = culc_rank(tag_feature, tag_feature)
img_rank_before_list = img_rank_before[np.eye(img_rank_before.shape[0])==0]
tag_rank_before_list = tag_rank_before[np.eye(tag_rank_before.shape[0])==0]
Plot(img_rank_before_list, tag_rank_before_list, number_of_font, "before")

# 埋め込み後の検索順位
img_rank_after = culc_rank(embedded_img_feature, embedded_img_feature)
tag_rank_after = culc_rank(embedded_tag_feature, embedded_tag_feature)
img_rank_after_list = img_rank_after[np.eye(img_rank_after.shape[0])==0]
tag_rank_after_list = tag_rank_after[np.eye(tag_rank_after.shape[0])==0]
Plot(img_rank_after_list, tag_rank_after_list, number_of_font, "after")