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
TAG_PREPROCESS = params.tag_preprocess
TAG_CLUSTER_PATH = params.tag_cluster_path

# 保存用ディレクトリの作成
SAVE_DIR = f'{EXPT}/clustering/save_cluster_tags/{TAG_PREPROCESS}'
os.makedirs(SAVE_DIR, exist_ok=True)

# パス，ラベル(クラスタID)の取得
_, tag_paths = utils.load_dataset_paths(DATASET)
tag_paths = np.asarray(tag_paths)
tag_cluster_id_marge = utils.load_hierarchical_clusterID(TAG_CLUSTER_PATH)


# 印象クラスタ別にタグのリストを保存
fontnames = utils.get_fontnames(DATASET)
tags = [utils.get_font_tags(tag_path) for tag_path in tag_paths]
write_rows = []
for i in range(len(tag_cluster_id_marge)):
    write_row = [tag_cluster_id_marge[i], fontnames[i]] + tags[i]
    write_rows.append(write_row)

sorted_rows = [x for _, x in sorted(zip(tag_cluster_id_marge, write_rows))]
utils.save_list_to_csv(sorted_rows, f'{SAVE_DIR}/{DATASET}.csv')
