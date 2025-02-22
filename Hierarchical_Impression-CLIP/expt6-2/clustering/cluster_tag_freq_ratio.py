'''
クラスタ別に各タグの全体に対する割合のヒストグラムとcsvを作成
'''
import numpy as np
import matplotlib.pyplot as plt
import copy

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

SAVE_DIR = f'{EXPT}/clustering/tag_freq_ratio/{TAG_PREPROCESS}/{DATASET}/{NUM_TAG_CLUSTERS}'
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

# タグリストを取得(train全体の頻度順)
tag_list = utils.get_tag_list()
tag_freq = {tag:0 for tag in tag_list}
# タグの頻度計算
for tag_path in tag_paths:
    for tag in utils.get_font_tags(tag_path):
        tag_freq[tag] += 1
# tag_listの並び替え
sorted_tag_list = sorted(tag_list, key=tag_freq.get, reverse=True)

# 各クラスターのタグの頻度を計算
frequency_list = []
for i in range(NUM_TAG_CLUSTERS):
    frequency_cluster_i = {tag:0 for tag in sorted_tag_list}
    for tags in tags_list_cluster[i]:
        for tag in tags:
            frequency_cluster_i[tag] += 1
    frequency_list.append(list(frequency_cluster_i.values()))

# クラスタ別に各タグの全体に対する割合を計算
frequency_rate_list = []
for i in range(NUM_TAG_CLUSTERS):
    frequency_rate = np.asarray(frequency_list[i])/np.asarray(list(tag_freq.values()))
    frequency_rate_list.append(frequency_rate)

# タグの割合をプロット
for i in range(NUM_TAG_CLUSTERS):
    ffig, ax = plt.subplots(figsize=(24, 6))
    plt.bar(sorted_tag_list, frequency_rate_list[i], edgecolor='black', width=1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=5.3, rotation=90)
    plt.xlabel(f'Tag')
    plt.ylabel('Frequency')
    plt.savefig(f'{SAVE_DIR}/cluster{i}.png', dpi=300, bbox_inches='tight')
    plt.close()

# csvに保存
frequency_rate_list = [row.tolist() for row in frequency_rate_list]
frequency_rate_list_to_write = copy.copy(frequency_rate_list)
for i, sublist in enumerate(frequency_rate_list_to_write):
    sublist.insert(0, f'cluster{i}')
first_row = [''] + sorted_tag_list
frequency_rate_list_to_write.insert(0, first_row)
utils.save_list_to_csv(frequency_rate_list_to_write, f'{SAVE_DIR}/tag_freq_ratio.csv')