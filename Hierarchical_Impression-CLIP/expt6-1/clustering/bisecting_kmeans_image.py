'''
画像特徴を標準化してbisecting kmeansでクラスタリング
'''

import torch
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from models import FontAutoencoder
from lib import utils
from lib import bisecting_kmeans
from lib import clustering_utils
from models import HierarchicalDataset


params = utils.get_parameters()
EXPT = params['expt']
FONTAUTOENCODER_PATH = params['fontautoencoder_path']
BATCH_SIZE = params['batch_size']
DATASET = params['dataset']
IMG_PREPROCESS = params['img_preprocess']
IMG_FEATURE_PATH = params['img_feature_path']
IMG_CLUSTER_PATH = params['img_cluster_path']

NUMBER_OF_CLUSTERS_MAX = 10
N_ITER = 100  # bisecting kmeansを異なる初期値で繰り返す回数

SAVE_DIR = os.path.dirname(IMG_CLUSTER_PATH)
os.makedirs(SAVE_DIR, exist_ok=True)

# 画像特徴の取得&標準化(train基準)
img_feature_path_train = f'{EXPT}/img_features/train.pth'
img_features_train = torch.load(img_feature_path_train)
scaler = StandardScaler()
standardizer = scaler.fit(img_features_train)
img_features = torch.load(IMG_FEATURE_PATH)
standardized_img_features = standardizer.transform(img_features)

for NUMBER_OF_CLUSTERS in range(2, NUMBER_OF_CLUSTERS_MAX+1): 
    # bisecting kmeans
    utils.fix_seed(6)
    best_inertia = np.inf
    for i in range(N_ITER):
        _, label = bisecting_kmeans.bisecting_kmeans(standardized_img_features, NUMBER_OF_CLUSTERS)
        inertia = bisecting_kmeans.calculate_inertia(standardized_img_features, label)
        # inertia最小のクラスタリング結果を使用
        if inertia < best_inertia:
            best_label = label
            best_inertia = inertia
        
    replaced_label = clustering_utils.replace_label(best_label)
    np.savez_compressed(f'{SAVE_DIR}/{NUMBER_OF_CLUSTERS}.npz', replaced_label)
    print(f'NUMBER_OF_CLUSTERS={NUMBER_OF_CLUSTERS}, best_inertia={best_inertia}')