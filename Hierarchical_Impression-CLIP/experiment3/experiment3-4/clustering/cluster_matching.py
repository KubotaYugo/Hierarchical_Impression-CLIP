import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


EXP = utils.EXP
DATASET = "train"
IMG_CLUSTER_PATH = f'{EXP}/clustering/{DATASET}/image_clusters.npz'
TAG_CLUSTER_PATH = f'{EXP}/clustering/{DATASET}/impression_clusters.npz'
SAVE_DIR = f'{EXP}/clustering/pair_frequency_heatmap'

os.makedirs(SAVE_DIR, exist_ok=True)

img_hierarchy = np.load(IMG_CLUSTER_PATH)["arr_0"].astype(np.int64)
tag_hierarchy = np.load(TAG_CLUSTER_PATH)["arr_0"].astype(np.int64)

# (1) 同じインデックスの要素のペアを作成
pairs = list(zip(img_hierarchy, tag_hierarchy))

# (2) ペアの頻度をカウント
pair_count = Counter(pairs)

# ヒートマップ用のカウント配列を作成 (0から9までの番号)
heatmap = np.zeros((10, 10))

# ペアの頻度をヒートマップ用の配列に反映
for (i, j), count in pair_count.items():
    heatmap[i][j] = count

# ヒートマップを保存
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap, annot=True, cmap="YlGnBu", fmt="g", cbar=True)
plt.xlabel("tag cluster id")
plt.ylabel("img cluster id")
plt.savefig(f'{SAVE_DIR}/{DATASET}_freq.png', dpi=300)
plt.close()

# 縦軸方向に正規化したヒートマップを作成
column_sums = heatmap.sum(axis=0)
normalized_heatmap_y = heatmap / column_sums[np.newaxis, :]
plt.figure(figsize=(8, 6))
sns.heatmap(normalized_heatmap_y, annot=True, cmap="YlGnBu", fmt=".2f", cbar=True)
plt.xlabel("tag cluster id")
plt.ylabel("img cluster id")
plt.savefig(f'{SAVE_DIR}/{DATASET}_normalized_y.png', dpi=300)
plt.close()

# 横軸方向に正規化したヒートマップを作成
row_sums = heatmap.sum(axis=1)
normalized_heatmap_x = heatmap / row_sums[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.heatmap(normalized_heatmap_x, annot=True, cmap="YlGnBu", fmt=".2f", cbar=True)
plt.xlabel("tag cluster id")
plt.ylabel("img cluster id")
plt.savefig(f'{SAVE_DIR}/{DATASET}_normalized_x.png', dpi=300)
plt.close()