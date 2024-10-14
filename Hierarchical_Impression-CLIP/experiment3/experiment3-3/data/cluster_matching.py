import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


IMG_HIERARCHY_PATH = utils.IMG_HIERARCHY_PATH
TAG_HIERARCHY_PATH = utils.TAG_HIERARCHY_PATH
EXP = utils.EXP

img_hierarchy = np.load(IMG_HIERARCHY_PATH)["arr_0"].astype(np.int64)
tag_hierarchy = np.load(TAG_HIERARCHY_PATH)["arr_0"].astype(np.int64)

# (1) 同じインデックスの要素のペアを作成
pairs = list(zip(img_hierarchy, tag_hierarchy))

# (2) ペアの頻度をカウント
pair_count = Counter(pairs)

# ヒートマップ用のカウント配列を作成 (0から9までの番号)
heatmap_data = np.zeros((10, 10))

# ペアの頻度をヒートマップ用の配列に反映
for (i, j), count in pair_count.items():
    heatmap_data[i][j] = count

# ヒートマップを表示
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt="g", cbar=True)
plt.title("Pair frequency heatmap")
plt.xlabel("tag_hierarchy")
plt.ylabel("img_hierarchy")
plt.savefig(f'{EXP}/Pair_frequency_heatmap.png', dpi=300)