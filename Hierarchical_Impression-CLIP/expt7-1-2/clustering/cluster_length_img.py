'''
各画像が属するクラスタの個数(階層の数)の分布をヒストグラムで可視化する
'''

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


# define constant
params = utils.get_parameters()
EXPT = params.expt
DATASET = params.dataset
IMG_CLUSTER_PATH = params.img_cluster_path

# 保存用ディレクトリの作成
SAVE_DIR = f'{EXPT}/clustering/cluster_length_imgs'
os.makedirs(SAVE_DIR, exist_ok=True)

# パス，ラベル(クラスタID)の取得 & 階層の数を計算
img_paths, _ = utils.load_dataset_paths(DATASET)
img_paths = np.asarray(img_paths)
img_cluster_id = np.load(IMG_CLUSTER_PATH)['arr_0'].astype(np.int64)
img_cluster_length = [np.sum(row==0)+np.sum(row==1) for row in img_cluster_id]
img_cluster_length = np.asarray(img_cluster_length)

# ヒストグラムの作成
categories = np.arange(img_cluster_id.shape[1]+1)
img_cluster_length_frequency = [np.sum(img_cluster_length==i) for i in range(img_cluster_id.shape[1]+1)]
plt.bar(categories, img_cluster_length_frequency, edgecolor='black', width=1)
plt.savefig(f'{SAVE_DIR}/{DATASET}.png', dpi=300, bbox_inches='tight')
plt.close()