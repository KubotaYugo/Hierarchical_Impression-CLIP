'''
各クラスタのタグの頻度のヒストグラムとcsvを作成
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
IMG_CLUSTER_PATH = params.img_cluster_path
TAG_CLUSTER_PATH = params.tag_cluster_path
NUM_IMG_CLUSTERS = params.num_img_clusters
NUM_TAG_CLUSTERS = params.num_tag_clusters
TAG_PREPROCESS = params.tag_preprocess

# ディレクトリの作成
SAVE_DIR = f'{EXPT}/clustering/tag_freq/{TAG_PREPROCESS}/{DATASET}/{NUM_TAG_CLUSTERS}'
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
for tag_path in tag_paths:
    for tag in utils.get_font_tags(tag_path):
        tag_freq[tag] += 1
sorted_tag_list = sorted(tag_list, key=tag_freq.get, reverse=True)

# 各クラスターのタグの頻度を計算
frequency_list = []
for i in range(NUM_TAG_CLUSTERS):
    frequency_cluster_i = {tag:0 for tag in sorted_tag_list}
    for tags in tags_list_cluster[i]:
        for tag in tags:
            frequency_cluster_i[tag] += 1
    frequency_list.append(list(frequency_cluster_i.values()))

# 各クラスターのタグの頻度をプロット
for i in range(NUM_TAG_CLUSTERS):
    fig, ax = plt.subplots(figsize=(24, 6))
    plt.bar(sorted_tag_list, frequency_list[i], edgecolor='black', width=1)
    plt.xticks(fontsize=5.3, rotation=90)
    plt.xlabel(f'Tag')
    plt.ylabel('Frequency')
    plt.savefig(f'{SAVE_DIR}/cluster{i}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 全フォントの印象タグの頻度
fig, ax = plt.subplots(figsize=(24, 6))
plt.bar(sorted_tag_list, tag_freq.values(), edgecolor='black', width=1)
plt.xticks(fontsize=5.3, rotation=90)
plt.xlabel(f'Tag')
plt.ylabel('Frequency')
plt.savefig(f'{SAVE_DIR}/total.png', dpi=300, bbox_inches='tight')
plt.close()

# csvに保存
frequency_list_to_write = copy.deepcopy(frequency_list)
for i, sublist in enumerate(frequency_list_to_write):
    sublist.insert(0, f'cluster{i}')
first_row = [''] + sorted_tag_list
last_row = ['total'] + list(tag_freq.values())
frequency_list_to_write.insert(0, first_row)
frequency_list_to_write.append(last_row)
utils.save_list_to_csv(frequency_list_to_write, f'{SAVE_DIR}/tag_freq.csv')