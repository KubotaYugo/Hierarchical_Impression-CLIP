import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from scipy.cluster.hierarchy import linkage, leaves_list

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
EXPT = params['expt']
DATASET = 'val'
IMG_CLUSTER_PATH = f'{EXPT}/clustering/clustering_img/{DATASET}/{params['num_img_clusters']}.npz'
TAG_CLUSTER_PATH = f'{EXPT}/clustering/clustering_tag/{DATASET}/{params['num_tag_clusters']}.npz'
SAVE_DIR = f'{EXPT}/clustering/pair_frequency_heatmap'

os.makedirs(SAVE_DIR, exist_ok=True)

img_hierarchy = np.load(IMG_CLUSTER_PATH)["arr_0"].astype(np.int64)
tag_hierarchy = np.load(TAG_CLUSTER_PATH)["arr_0"].astype(np.int64)

# 同じインデックスの要素のペアを作成
pairs = list(zip(img_hierarchy, tag_hierarchy))
# ペアの頻度をカウント
pair_count = Counter(pairs)
# ペアの頻度をヒートマップ用の配列に反映
heatmap = np.zeros((10, 10))
for (i, j), count in pair_count.items():
    heatmap[i][j] = count

# ヒートマップを保存
save_heatmap(heatmap, 'g', f'{SAVE_DIR}/{DATASET}_freq.png')

# 縦軸方向に正規化したヒートマップを保存
column_sums = heatmap.sum(axis=0)
normalized_heatmap_y = heatmap / column_sums[np.newaxis, :]
save_heatmap(normalized_heatmap_y, '.2f', f'{SAVE_DIR}/{DATASET}_normalized_y.png')

# 横軸方向に正規化したヒートマップを保存
row_sums = heatmap.sum(axis=1)
normalized_heatmap_x = heatmap / row_sums[:, np.newaxis]
save_heatmap(normalized_heatmap_x, '.2f', f'{SAVE_DIR}/{DATASET}_normalized_x.png')