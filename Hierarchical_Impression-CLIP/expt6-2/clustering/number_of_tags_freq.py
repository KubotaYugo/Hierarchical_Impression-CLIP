'''
各クラスタのタグの個数の頻度のヒストグラムとcsvを作成
'''
import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import Counter

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

SAVE_DIR = f'{EXPT}/clustering/number_of_tags_frequency/{TAG_PREPROCESS}/{DATASET}/{NUM_TAG_CLUSTERS}'
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

# クラスタ別に印象タグの個数の頻度を計算
frequency_base = {x+1:0 for x in range(NUM_TAG_CLUSTERS)}
number_of_tags_frequency_list = []
for i in range(NUM_TAG_CLUSTERS):
    # 個数の頻度を計算
    number_of_tags_cluster_i = [len(tags) for tags in tags_list_cluster[i]]
    increment = Counter(number_of_tags_cluster_i)
    frequency = copy.copy(frequency_base)
    frequency.update(increment)
    frequency_sorted = [x for _, x in sorted(zip(frequency.keys(), frequency.values()))]
    number_of_tags_frequency_list.append(frequency_sorted)

# フォント1つあたりのタグの個数の最大値を計算
number_of_tags_max = max([len(utils.get_font_tags(tag_path)) for tag_path in tag_paths])

# クラスタ別に印象タグの個数の頻度をプロット
categories = [x+1 for x in range(NUM_TAG_CLUSTERS)]
for i in range(NUM_TAG_CLUSTERS):
    plt.bar(categories, number_of_tags_frequency_list[i], edgecolor='black', width=1)
    plt.xticks(ticks=[x+1 for x in range(number_of_tags_max)], labels=categories)
    plt.xlim(0, number_of_tags_max+1)
    plt.xlabel(f'Number of tags in impression cluster{i}')
    plt.ylabel('Frequency')
    plt.savefig(f'{SAVE_DIR}/cluster{i}.png', dpi=300, bbox_inches='tight')
    plt.close()

# トータルのタグの個数の頻度をプロット
number_of_tags_frequency_list_temp = np.asarray(number_of_tags_frequency_list) 
plt.bar(categories, np.sum(number_of_tags_frequency_list_temp, axis=0), edgecolor='black', width=1)
plt.xticks(ticks=[x+1 for x in range(number_of_tags_max)], labels=categories)
plt.xlim(0, number_of_tags_max+1)
plt.xlabel(f'Number of tags in impression cluster{i}')
plt.ylabel('Frequency')
plt.savefig(f'{SAVE_DIR}/total.png', dpi=300, bbox_inches='tight')
plt.close()

# クラスタ別の印象タグの個数の頻度をcsvで保存
write_rows = copy.copy(number_of_tags_frequency_list)
for i, sublist in enumerate(write_rows):
    sublist.insert(0, f'cluster{i}')
first_row = [''] + [x+1 for x in range(number_of_tags_max)]
write_rows.insert(0, first_row)
utils.save_list_to_csv(write_rows, f'{SAVE_DIR}/number_of_tags_frequency.csv')