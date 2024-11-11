import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


def save_heatmap(data, fmt, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(data, annot=True, cmap="YlGnBu", fmt=fmt, cbar=True)
    plt.xlabel("tag cluster id")
    plt.ylabel("img cluster id")
    plt.savefig(filename, dpi=300)
    plt.close()


params = utils.get_parameters()
EXPT = params.expt
DATASET = params.dataset
TAG_PREPROCESS = params.tag_preprocess
NUM_IMG_CLUSTERS = params.num_img_clusters
NUM_TAG_CLUSTERS = params.num_tag_clusters
IMG_CLUSTER_PATH = params.img_cluster_path
TAG_CLUSTER_PATH = params.tag_cluster_path

SAVE_DIR = f'{EXPT}/clustering/pair_frequency_heatmap/{TAG_PREPROCESS}/{DATASET}/{NUM_IMG_CLUSTERS}_{NUM_TAG_CLUSTERS}'
os.makedirs(SAVE_DIR, exist_ok=True)

img_cluster = np.load(IMG_CLUSTER_PATH)["arr_0"].astype(np.int64)
tag_cluster = np.load(TAG_CLUSTER_PATH)["arr_0"].astype(np.int64)

pairs = list(zip(img_cluster, tag_cluster))     # 同じインデックスの要素のペアを作成
pair_count = Counter(pairs)                     # ペアの頻度をカウント
heatmap = np.zeros((10, 10))                    # ペアの頻度をヒートマップ用の配列に反映
for (i, j), count in pair_count.items():
    heatmap[i][j] = count


# ヒートマップを保存
save_heatmap(heatmap, 'g', f'{SAVE_DIR}/freq.png')

# 縦軸方向に正規化したヒートマップを保存
column_sums = heatmap.sum(axis=0)
normalized_heatmap_y = heatmap / column_sums[np.newaxis, :]
save_heatmap(normalized_heatmap_y, '.2f', f'{SAVE_DIR}/normalized_y.png')

# 横軸方向に正規化したヒートマップを保存
row_sums = heatmap.sum(axis=1)
normalized_heatmap_x = heatmap / row_sums[:, np.newaxis]
save_heatmap(normalized_heatmap_x, '.2f', f'{SAVE_DIR}/normalized_x.png')