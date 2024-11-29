'''
クラスタの印象タグをcsvに保存
'''
import numpy as np

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


# define constant
params = utils.get_parameters()
EXPT = params.expt
DATASET = params.dataset
TAG_CLUSTER_PATH = params.tag_cluster_path
NUM_TAG_CLUSTERS = params.num_tag_clusters
TAG_PREPROCESS = params.tag_preprocess

SAVE_DIR = f'{EXPT}/clustering/cluster_tags/{TAG_PREPROCESS}/{DATASET}/{NUM_TAG_CLUSTERS}'
os.makedirs(SAVE_DIR, exist_ok=True)

# パス，ラベル(クラスタID)の取得
_, tag_paths = utils.load_dataset_paths(DATASET)
tag_paths = np.asarray(tag_paths)
tag_cluster_id = np.load(TAG_CLUSTER_PATH)['arr_0'].astype(np.int64)

# クラスタ別のタグのリストを作成
tags_list_cluster = []
for i in range(NUM_TAG_CLUSTERS):
    tag_path_cluster_i = tag_paths[tag_cluster_id==i]
    tags_list = [utils.get_font_tags(tag_path) for tag_path in tag_path_cluster_i]
    tags_list_cluster.append(tags_list)

# クラスタ別に印象タグを保存
for i in range(NUM_TAG_CLUSTERS):
    utils.save_list_to_csv(tags_list_cluster[i], f'{SAVE_DIR}/cluster{i}.csv')