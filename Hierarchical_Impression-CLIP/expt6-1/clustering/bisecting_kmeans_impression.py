'''
印象特徴を標準化してbisecting kmeansでクラスタリング
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
from lib import clustering_utils

params = utils.get_parameters()
EXPT = params['expt']
FONTAUTOENCODER_PATH = params['fontautoencoder_path']
BATCH_SIZE = params['batch_size']
DATASET = params['dataset']
TAG_PREPROCESS = params['tag_preprocess']
TAG_FEATURE_PATH = params['tag_feature_path']
TAG_CLUSTER_PATH = params['tag_cluster_path']

NUMBER_OF_CLUSTERS_MAX = 10
N_ITER = 100  # bisecting kmeansを異なる初期値で繰り返す回数

SAVE_DIR = os.path.dirname(TAG_CLUSTER_PATH)
os.makedirs(SAVE_DIR, exist_ok=True)

# 印象特徴の取得&標準化(train基準)
tag_feature_path_train = f'{EXPT}/tag_features/train/{TAG_PREPROCESS}.pth'
tag_features_train = torch.load(tag_feature_path_train)
scaler = StandardScaler()
standardizer = scaler.fit(tag_features_train)
tag_features = torch.load(TAG_FEATURE_PATH)
standardized_tag_features = standardizer.transform(tag_features)

for NUMBER_OF_CLUSTERS in range(2, NUMBER_OF_CLUSTERS_MAX+1): 
    # bisecting kmeans
    utils.fix_seed(7)
    best_inertia = np.inf
    for i in range(N_ITER):
        _, label = bisecting_kmeans.bisecting_kmeans(standardized_tag_features, NUMBER_OF_CLUSTERS)
        inertia = bisecting_kmeans.calculate_inertia(standardized_tag_features, label)
        # inertia最小のクラスタリング結果を使用
        if inertia < best_inertia:
            best_label = label
            best_inertia = inertia

    replaced_label = clustering_utils.replace_label(best_label)
    np.savez_compressed(f'{SAVE_DIR}/{NUMBER_OF_CLUSTERS}.npz', replaced_label)
    print(f'NUMBER_OF_CLUSTERS={NUMBER_OF_CLUSTERS}, best_inertia={best_inertia}')