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
IMG_CLUSTER_PATH = params.img_cluster_path
TAG_CLUSTER_PATH = params.tag_cluster_path
NUM_IMG_CLUSTERS = params.num_img_clusters
NUM_TAG_CLUSTERS = params.num_tag_clusters
TAG_PREPROCESS = params.tag_preprocess
BASE_SAVE_DIR = f'{EXPT}/clustering/cluster_tags_new/num_img_cluster={NUM_IMG_CLUSTERS}_num_tag_cluster={NUM_TAG_CLUSTERS}'

# パス，ラベル(クラスタID)の取得
_, tag_paths = utils.load_dataset_paths(DATASET)
tag_paths = np.asarray(tag_paths)
img_cluster_id = np.load(IMG_CLUSTER_PATH)['arr_0'].astype(np.int64)
tag_cluster_id = np.load(TAG_CLUSTER_PATH)['arr_0'].astype(np.int64)

# 印象クラスタ別にタグのリストを保存 (フォント名も)
SAVE_DIR = f'{BASE_SAVE_DIR}/{TAG_PREPROCESS}/tag_cluster/{DATASET}'
os.makedirs(SAVE_DIR, exist_ok=True)
for i in range(NUM_TAG_CLUSTERS):
    for j in range(NUM_IMG_CLUSTERS):
        mask = (tag_cluster_id==i)*(img_cluster_id==j)
        tag_path_cluster_ij = tag_paths[mask]
        tags_list = [[os.path.splitext(os.path.basename(tag_path))[0]]+utils.get_font_tags(tag_path) for tag_path in tag_path_cluster_ij]
        utils.save_list_to_csv(tags_list, f'{SAVE_DIR}/[tag_cluster, img_cluster]=[{i+1}, {j+1}].csv')

# 画像クラスタ別にタグのリストを保存 (フォント名も)
SAVE_DIR = f'{BASE_SAVE_DIR}/{TAG_PREPROCESS}/img_cluster/{DATASET}'
os.makedirs(SAVE_DIR, exist_ok=True)
for i in range(NUM_IMG_CLUSTERS):
    for j in range(NUM_TAG_CLUSTERS):
        mask = (img_cluster_id==i)*(tag_cluster_id==j)
        tag_path_cluster_ij = tag_paths[mask]
        tags_list = [[os.path.splitext(os.path.basename(tag_path))[0]]+utils.get_font_tags(tag_path) for tag_path in tag_path_cluster_ij]
        utils.save_list_to_csv(tags_list, f'{SAVE_DIR}/[img_cluster, tag_cluster]=[{i+1}, {j+1}].csv')