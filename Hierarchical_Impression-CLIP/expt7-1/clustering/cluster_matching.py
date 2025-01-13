'''
特定の画像の階層と印象の階層でどんな風に対応しているかを見る
'''


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


def save_heatmap(data, labels, fmt, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data, xticklabels=labels[1], yticklabels=labels[0], annot=True, cmap="YlGnBu", fmt=fmt, cbar=True, ax=ax, annot_kws={"size": 6})
    plt.xlabel("tag cluster id")
    plt.ylabel("img cluster id")
    ax.figure.tight_layout()
    ax.set_aspect('equal')
    plt.savefig(filename, dpi=300)
    plt.close()


# ハイパラ
params = utils.get_parameters()
EXPT = params.expt
DATASET = params.dataset
TAG_PREPROCESS = params.tag_preprocess
IMG_CLUSTER_PATH = params.img_cluster_path
TAG_CLUSTER_PATH = params.tag_cluster_path

# 保存用ディレクトリの作成
SAVE_DIR = f'{EXPT}/clustering/pair_frequency_heatmap/{TAG_PREPROCESS}/{DATASET}'
os.makedirs(SAVE_DIR, exist_ok=True)

for CLUSTER_LENGTH in range(2, 6):
    IMG_CLUSTER_LENGTH = CLUSTER_LENGTH
    TAG_CLUSTER_LENGTH = CLUSTER_LENGTH
    # クラスタIDの読み込み & 刈り取り
    img_cluster_id = utils.load_hierarchical_clusterID(IMG_CLUSTER_PATH, IMG_CLUSTER_LENGTH)
    tag_cluster_id = utils.load_hierarchical_clusterID(TAG_CLUSTER_PATH, TAG_CLUSTER_LENGTH)

    # ペアの頻度をヒートマップに
    img_labels = [f"{i:0{IMG_CLUSTER_LENGTH}b}" for i in range(pow(2,IMG_CLUSTER_LENGTH))]
    tag_labels = [f"{i:0{TAG_CLUSTER_LENGTH}b}" for i in range(pow(2,TAG_CLUSTER_LENGTH))]
    labels = [img_labels, tag_labels]

    heatmap = np.zeros((len(img_labels), len(tag_labels)))
    for i in range(len(img_labels)):
        for j in range(len(tag_labels)):
            heatmap[i][j] = np.sum((img_cluster_id==img_labels[i]) * (tag_cluster_id==tag_labels[j]))

    # ヒートマップを保存
    save_heatmap(heatmap, labels, 'g', f'{SAVE_DIR}/({IMG_CLUSTER_LENGTH},{TAG_CLUSTER_LENGTH})_freq.png')

    # 縦軸方向に正規化したヒートマップを保存
    column_sums = heatmap.sum(axis=0)
    normalized_heatmap_y = heatmap / column_sums[np.newaxis, :] * 100
    save_heatmap(normalized_heatmap_y, labels, '.0f', f'{SAVE_DIR}/({IMG_CLUSTER_LENGTH},{TAG_CLUSTER_LENGTH})_normalized_y.png')

    # 横軸方向に正規化したヒートマップを保存
    row_sums = heatmap.sum(axis=1)
    normalized_heatmap_x = heatmap / row_sums[:, np.newaxis] * 100
    save_heatmap(normalized_heatmap_x, labels, '.0f', f'{SAVE_DIR}/({IMG_CLUSTER_LENGTH},{TAG_CLUSTER_LENGTH})_normalized_x.png')