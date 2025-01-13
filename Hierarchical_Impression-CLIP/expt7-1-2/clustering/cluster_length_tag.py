'''
各印象が属するクラスタの個数(階層の数)の分布をヒストグラムで可視化する
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
TAG_PREPROCESS = params.tag_preprocess
TAG_CLUSTER_PATH = params.tag_cluster_path
DATASET = params.dataset

# 保存用ディレクトリの作成
SAVE_DIR = f'{EXPT}/clustering/cluster_length_tags/{TAG_PREPROCESS}'
os.makedirs(SAVE_DIR, exist_ok=True)

# パス，ラベル(クラスタID)の取得 & 階層の数を計算
_, tag_paths = utils.load_dataset_paths(DATASET)
tag_paths = np.asarray(tag_paths)
tag_cluster_id = np.load(TAG_CLUSTER_PATH)['arr_0'].astype(np.int64)
tag_cluster_length = [np.sum(row==0)+np.sum(row==1) for row in tag_cluster_id]
tag_cluster_length = np.asarray(tag_cluster_length)

# ヒストグラムの作成
categories = np.arange(tag_cluster_id.shape[1]+1)
tag_cluster_length_frequency = [np.sum(tag_cluster_length==i) for i in range(tag_cluster_id.shape[1]+1)]
plt.bar(categories, tag_cluster_length_frequency, edgecolor='black', width=1)
plt.savefig(f'{SAVE_DIR}/{DATASET}.png', dpi=300, bbox_inches='tight')
plt.close()