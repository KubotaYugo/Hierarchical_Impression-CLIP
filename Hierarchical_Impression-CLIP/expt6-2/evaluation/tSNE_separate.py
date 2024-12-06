'''
以下の4つのプロットを作成
画像特徴だけで画像クラスタで色分けしたプロット
画像特徴だけで印象クラスタで色分けしたプロット
印象特徴だけで画像クラスタで色分けしたプロット
印象特徴だけで印象クラスタで色分けしたプロット
'''
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils

def plot_PC1_PC2(X, Y, labels, feature, color):
    plt.subplots(figsize=(6.4*1.5, 4.8*1.5))
    patches = [mpatches.Patch(color=plt.cm.tab10(i), label=f'cluster{i}') for i in range(10)]
    plt.scatter(X, Y, c=plt.cm.tab10(np.asarray(labels, dtype=np.int64)), alpha=0.8, edgecolors='w', linewidths=0.1, s=5)
    plt.legend(handles=patches)
    plt.savefig(f'{SAVE_DIR}/feature={feature}, color={color}.png', bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()


# define constant
params = utils.get_parameters()
DATASET = params.dataset
BASE_DIR = params.base_dir
IMG_CLUSTER_PATH = params.img_cluster_path
TAG_CLUSTER_PATH = params.tag_cluster_path

# ディレクトリの作成
SAVE_DIR = f'{BASE_DIR}/tSNE_separate/{DATASET}'
os.makedirs(f'{SAVE_DIR}', exist_ok=True)

# クラスタIDの読み込み
img_cluster_id = np.load(IMG_CLUSTER_PATH)["arr_0"].astype(np.int64)
tag_cluster_id = np.load(TAG_CLUSTER_PATH)["arr_0"].astype(np.int64)

# tSNE特徴の読み込み
tSNE_feature_filename = f'{BASE_DIR}/tSNE/{DATASET}/tSNE_feature.npz'
tSNE_embedding = np.load(tSNE_feature_filename)['arr_0']
x = tSNE_embedding[:,0]
y = tSNE_embedding[:,1]
t = t = int(len(x)/2)
img_x = x[:t]
img_y = y[:t]
tag_x = x[t:]
tag_y = y[t:]

# 画像特徴だけで画像クラスタで色分けしたプロット
plot_PC1_PC2(img_x, img_y, img_cluster_id, 'img', 'img')
plot_PC1_PC2(img_x, img_y, tag_cluster_id, 'img', 'tag')
plot_PC1_PC2(tag_x, tag_y, img_cluster_id, 'tag', 'img')
plot_PC1_PC2(tag_x, tag_y, tag_cluster_id, 'tag', 'tag')