'''
各階層のクラスタの大きさの分布を見る
'''

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils

# =======================================
# 書いている途中
# =======================================

# define constant
params = utils.get_parameters()
EXPT = params.expt
DATASET = params.dataset
IMG_CLUSTER_PATH = params.img_cluster_path

# 保存用ディレクトリの作成
SAVE_DIR = f'{EXPT}/clustering/cluster_size_imgs'
os.makedirs(SAVE_DIR, exist_ok=True)

# パス，ラベル(クラスタID)の取得
img_cluster_id = utils.load_hierarchical_clusterID(IMG_CLUSTER_PATH)
img_cluster_id = [[int(char) for char in string] for string in img_cluster_id]
img_cluster_id = np.asarray(img_cluster_id)

# 各階層におけるクラスタの大きさの計算
for layer in range(img_cluster_id.shape[1]):
    img_cluster_id_layer = img_cluster_id[:,:layer+1]
    unique_ids = np.unique(img_cluster_id_layer)
    cluster_size_layer = []
    for unique_id in unique_ids:
        cluste_size_unique_id = np.sum(img_cluster_id_layer==unique_id)
        cluster_size_layer.append(cluste_size_unique_id)
