'''
共埋め込み後の画像特徴と印象特徴の可視化
(クラスタの学習を入れなかった場合)
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


# define constant
params = utils.get_parameters()
DATASET = params.dataset
BASE_DIR = params.base_dir
IMG_CLUSTER_PATH = params.img_cluster_path
TAG_CLUSTER_PATH = params.tag_cluster_path
DEPTH = 4   # クラスタのラベルをどの階層まで見るか
N = 5       # 可視化する次元の数

# ディレクトリの作成
SAVE_DIR = f'{BASE_DIR}/PCA_baseline_cluster/{DATASET}'
os.makedirs(f'{SAVE_DIR}', exist_ok=True)

# ラベル(クラスタID)の取得
img_cluster_id = utils.load_hierarchical_clusterID(IMG_CLUSTER_PATH, DEPTH)
tag_cluster_id = utils.load_hierarchical_clusterID(TAG_CLUSTER_PATH, DEPTH)

# PCA特徴の読み込み
PCA_feature_filename = f'{BASE_DIR}/PCA_baseline/{DATASET}/PCA_feature.npz'
embedding = np.load(PCA_feature_filename)['arr_0']
embedding_img = embedding[:int(len(embedding)/2)]
embedding_tag = embedding[int(len(embedding)/2):]

# 主成分方向の分布
def plot(embedding, cluster_id, filename):
    df = pd.DataFrame(embedding[:, :N])
    df.columns = [f'PC{i+1}' for i in range(N)]
    df['cluster'] = cluster_id
    palette = 'tab10' if len(cluster_id[0])<4 else 'tab20'
    sns.pairplot(df, hue='cluster', palette=palette, hue_order=np.unique(cluster_id), plot_kws={'s':2})
    plt.savefig(f'{SAVE_DIR}/{filename}.png', bbox_inches='tight', dpi=500)
    # plt.show()
    plt.close()

plot(embedding_img, img_cluster_id, f'N={N}_D={DEPTH}_img')
plot(embedding_tag, tag_cluster_id, f'N={N}_D={DEPTH}_tag')