'''
クラスタ内の印象がもつ印象タグをcsvに保存
クラスタごとに,個数別の頻度,種類別の頻度をプロット
'''
import numpy as np
import matplotlib.pyplot as plt
import copy
import csv
from collections import Counter

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


def save_list_to_csv(data_list, output_path='output.csv'):
    """
    リストの各要素を1行としてCSVファイルに保存する。
    
    :param data_list: リスト (各要素が1行として保存される)
    :param output_path: 出力CSVファイルのパス
    """
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_list)

# define constant
params = utils.get_parameters()
EXPT = params['expt']
DATASET = params['dataset']
TAG_CLUSTER_PATH = params['tag_cluster_path']
NUM_TAG_CLUSTERS = params['num_tag_clusters']
TAG_PREPROCESS = params['tag_preprocess']

# パス，ラベル(クラスタID)の取得
_, tag_paths = utils.load_dataset_paths(DATASET)
tag_paths = np.asarray(tag_paths)
tag_cluster_id = np.load(TAG_CLUSTER_PATH)["arr_0"].astype(np.int64)

# クラスタ別にタグのリストを取得
tags_list_cluster = []
for i in range(NUM_TAG_CLUSTERS):
    tag_path_cluster_i = tag_paths[tag_cluster_id==i]
    tags_list = [utils.get_font_tags(tag_path) for tag_path in tag_path_cluster_i]
    tags_list_cluster.append(tags_list)

# クラスタ別に印象タグを保存
SAVE_DIR = f"{EXPT}/clustering/save_cluster_tags/{DATASET}/{TAG_PREPROCESS}/{NUM_TAG_CLUSTERS}"
for i in range(NUM_TAG_CLUSTERS):
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_list_to_csv(tags_list_cluster[i], f'{SAVE_DIR}/cluster{i}.csv')



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

# フォント1つあたりのタグの個数の最大
number_of_tags_max = max([len(utils.get_font_tags(tag_path)) for tag_path in tag_paths])

# クラスタ別に印象タグの個数の頻度をプロット
SAVE_DIR = f"{EXPT}/clustering/number_of_tags_cluster/{DATASET}/{TAG_PREPROCESS}/{NUM_TAG_CLUSTERS}"
os.makedirs(SAVE_DIR, exist_ok=True)
categories = [x+1 for x in range(NUM_TAG_CLUSTERS)]
for i in range(NUM_TAG_CLUSTERS):
    plt.bar(categories, number_of_tags_frequency_list[i], edgecolor='black', width=1)
    plt.xticks(ticks=[x+1 for x in range(number_of_tags_max)], labels=categories)
    plt.xlim(0, number_of_tags_max+1)
    plt.xlabel(f'Number of tags in impression cluster{i}')
    plt.ylabel('Frequency')
    plt.savefig(f'{SAVE_DIR}/cluster{i}.png', dpi=500, bbox_inches='tight')
    plt.close()

# クラスタ別の印象タグの個数の頻度をcsvで保存
write_rows = copy.copy(number_of_tags_frequency_list)
for i, sublist in enumerate(write_rows):
    sublist.insert(0, f'cluster{i}')
first_row = [''] + [x+1 for x in range(number_of_tags_max)]
write_rows.insert(0, first_row)
save_list_to_csv(write_rows, f'{SAVE_DIR}/number_of_tags_frequency.csv')



# タグリストを取得(train全体の頻度順)
tag_list = list(utils.get_tag_list())
tag_freq = {tag:0 for tag in tag_list}
for tag_path in tag_paths:
    for tag in utils.get_font_tags(tag_path):
        tag_freq[tag] += 1
sorted_tag_list = sorted(tag_list, key=tag_freq.get, reverse=True)

# 各クラスターのタグの種類別の頻度を計算
frequency_list = []
for i in range(NUM_TAG_CLUSTERS):
    frequency_cluster_i = {tag:0 for tag in sorted_tag_list}
    for tags in tags_list_cluster[i]:
        for tag in tags:
            frequency_cluster_i[tag] += 1
    frequency_list.append(list(frequency_cluster_i.values()))

# タグの種類別の頻度をプロット
SAVE_DIR = f"{EXPT}/clustering/frequency_of_tags_cluster/{DATASET}/{TAG_PREPROCESS}/{NUM_TAG_CLUSTERS}"
os.makedirs(SAVE_DIR, exist_ok=True)
for i in range(NUM_TAG_CLUSTERS):
    fig, ax = plt.subplots(figsize=(24, 6))
    plt.bar(sorted_tag_list, frequency_list[i], edgecolor='black', width=1)
    plt.xticks(fontsize=5.3, rotation=90)
    plt.xlabel(f'Tag')
    plt.ylabel('Frequency')
    plt.savefig(f'{SAVE_DIR}/cluster{i}.png', dpi=500, bbox_inches='tight')
    plt.close()

# train内全フォントの印象タグの頻度
fig, ax = plt.subplots(figsize=(24, 6))
plt.bar(sorted_tag_list, tag_freq.values(), edgecolor='black', width=1)
plt.xticks(fontsize=5.3, rotation=90)
plt.xlabel(f'Tag')
plt.ylabel('Frequency')
plt.savefig(f'{SAVE_DIR}/total.png', dpi=500, bbox_inches='tight')
plt.close()

# csvに保存
frequency_list_to_write = copy.deepcopy(frequency_list)
for i, sublist in enumerate(frequency_list_to_write):
    sublist.insert(0, f'cluster{i}')
first_row = [''] + sorted_tag_list
last_row = ['total'] + list(tag_freq.values())
frequency_list_to_write.insert(0, first_row)
frequency_list_to_write.append(last_row)
save_list_to_csv(frequency_list_to_write, f'{SAVE_DIR}/cluster_frequency_of_tags.csv')



# タグの割合をプロット
frequency_rate_list = []
for i in range(NUM_TAG_CLUSTERS):
    frequency_rate = np.asarray(frequency_list[i])/np.asarray(list(tag_freq.values()))
    frequency_rate_list.append(frequency_rate)

# タグの割合を保存
SAVE_DIR = f"{EXPT}/clustering/cluster_frequency_rate_of_tags"
os.makedirs(SAVE_DIR, exist_ok=True)
for i in range(NUM_TAG_CLUSTERS):
    ffig, ax = plt.subplots(figsize=(24, 6))
    plt.bar(sorted_tag_list, frequency_rate_list[i], edgecolor='black', width=1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=5.3, rotation=90)
    plt.xlabel(f'Tag')
    plt.ylabel('Frequency')
    plt.savefig(f'{SAVE_DIR}/cluster{i}.png', dpi=500, bbox_inches='tight')
    plt.close()

# csvに保存
frequency_rate_list = [row.tolist() for row in frequency_rate_list]
frequency_rate_list_to_write = copy.copy(frequency_rate_list)
for i, sublist in enumerate(frequency_rate_list_to_write):
    sublist.insert(0, f'cluster{i}')
first_row = [''] + sorted_tag_list
frequency_rate_list_to_write.insert(0, first_row)
save_list_to_csv(frequency_rate_list_to_write, f'{SAVE_DIR}/cluster_frequency_rate_of_tags.csv')