'''
印象特徴を標準化し, bisecting kmeansでクラスタリング
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
TAG_PREPROCESS = params.tag_preprocess
TAG_FEATURE_PATH = params.tag_feature_path
TAG_CLUSTER_PATH = params.tag_cluster_path

# 保存用ディレクトリの作成
SAVE_DIR = os.path.dirname(TAG_CLUSTER_PATH)
os.makedirs(SAVE_DIR, exist_ok=True)

# 印象特徴の取得&標準化
tag_feature = torch.load(TAG_FEATURE_PATH).to("cpu").detach().numpy()
scaler = StandardScaler()
standardizer = scaler.fit(tag_feature)
standardized_tag_feature = standardizer.transform(tag_feature)

# 再帰的に2分割
indexes = np.arange(len(standardized_tag_feature))
cluster_tree = recursive_kmeans.recursive_kmeans(indexes, standardized_tag_feature)

# 各インデックスのデータまでのパス(=ラベル)を取得
cluster_paths = [recursive_kmeans.get_cluster_path(cluster_tree, index) for index in range(len(standardized_tag_feature))]
padded_cluster_paths = recursive_kmeans.pad_array(cluster_paths)
np.savez_compressed(TAG_CLUSTER_PATH, padded_cluster_paths)