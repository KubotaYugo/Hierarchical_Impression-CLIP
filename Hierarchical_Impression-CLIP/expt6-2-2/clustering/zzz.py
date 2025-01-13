'''
画像特徴を標準化し, kmeansでクラスタ数=データ数となるまで再帰的に2分割
(全く同じ画像は無理に2分割せずに, 同じクラスタにする)
'''

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import recursive_kmeans

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


# define constant
params = utils.get_parameters()
DATASET = params.dataset
IMG_FEATURE_PATH = params.img_feature_path
IMG_CLUSTER_PATH = params.img_cluster_path

# 画像特徴の取得 & 標準化
img_feature = torch.load(IMG_FEATURE_PATH).to("cpu").detach().numpy()
scaler = StandardScaler()
standardizer = scaler.fit(img_feature)
standardized_img_feature = standardizer.transform(img_feature)

# 再帰的に2分割
indexes = np.arange(len(standardized_img_feature))
cluster_tree = recursive_kmeans.recursive_kmeans(indexes, standardized_img_feature)

# 各インデックスのデータまでのパス(=ラベル)を取得
cluster_paths = [recursive_kmeans.get_cluster_path(cluster_tree, index) for index in range(len(standardized_img_feature))]
padded_cluster_paths = recursive_kmeans.pad_array(cluster_paths)
np.savez_compressed(IMG_CLUSTER_PATH, padded_cluster_paths)