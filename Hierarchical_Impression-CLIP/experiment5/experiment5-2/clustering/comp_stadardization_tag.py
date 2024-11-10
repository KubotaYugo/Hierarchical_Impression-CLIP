'''
標準化せずにクラスタリングした結果と標準化してクラスタリングした結果を比べる(印象)
クラスタ数は10で比較
'''
import numpy as np
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils

EXP = utils.EXP
DATASET = 'train'
SAVE_DIR = f'{EXP}/clustering/comp_standardization/{DATASET}'

TAG_CLUSTER_PATH = f'{EXP}/clustering_tag/{DATASET}/10.npz'
tag_cluster_id = np.load(TAG_CLUSTER_PATH)["arr_0"].astype(np.int64)
TAG_CLUSTER_PATH = f'{EXP}/clustering_standardized_tag/{DATASET}/10.npz'
tag_cluster_id_standardized = np.load(TAG_CLUSTER_PATH)["arr_0"].astype(np.int64)

os.makedirs(SAVE_DIR, exist_ok=True)

pairs = list(zip(tag_cluster_id, tag_cluster_id_standardized))
pair_count = Counter(pairs)

# ヒートマップ用のカウント配列を作成 (0から9までの番号)
heatmap = np.zeros((10, 10))

# ペアの頻度をヒートマップ用の配列に反映
for (i, j), count in pair_count.items():
    heatmap[i][j] = count

# ヒートマップを保存
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap, annot=True, cmap="YlGnBu", fmt="g", cbar=True)
plt.xlabel("tag standardized")
plt.ylabel("tag")
plt.savefig(f'{SAVE_DIR}/tag.png', dpi=300)
plt.close()