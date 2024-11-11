'''
印象特徴を標準化し, bisecting kmeansでクラスタリング
'''

import torch
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils
from lib import bisecting_kmeans


# ハイパラ
params = utils.get_parameters()
TAG_PREPROCESS = params.tag_preprocess
TAG_FEATURE_PATH = params.tag_feature_path
TAG_CLUSTER_PATH = params.tag_cluster_path

NUMBER_OF_CLUSTERS_MAX = 10
N_ITER = 100  # bisecting kmeansを異なる初期値で繰り返す回数

# 保存用ディレクトリの作成
SAVE_DIR = os.path.dirname(TAG_CLUSTER_PATH)
os.makedirs(SAVE_DIR, exist_ok=True)

# 印象特徴の取得&標準化
tag_feature = torch.load(TAG_FEATURE_PATH).to("cpu").detach().numpy()
scaler = StandardScaler()
standardizer = scaler.fit(tag_feature)
standardized_tag_feature = standardizer.transform(tag_feature)

# クラスタリング
for NUMBER_OF_CLUSTERS in range(2, NUMBER_OF_CLUSTERS_MAX+1): 
    # bisecting kmeans
    utils.fix_seed(7)
    best_inertia = np.inf
    for i in range(N_ITER):
        _, label = bisecting_kmeans.bisecting_kmeans(standardized_tag_feature, NUMBER_OF_CLUSTERS)
        inertia = bisecting_kmeans.calculate_inertia(standardized_tag_feature, label)
        # inertia最小のクラスタリング結果を保存
        if inertia < best_inertia:
            best_label = label
            best_inertia = inertia

    replaced_label = bisecting_kmeans.replace_label(best_label)
    np.savez_compressed(f'{SAVE_DIR}/{NUMBER_OF_CLUSTERS}.npz', replaced_label)
    print(f'NUMBER_OF_CLUSTERS={NUMBER_OF_CLUSTERS}, best_inertia={best_inertia}')