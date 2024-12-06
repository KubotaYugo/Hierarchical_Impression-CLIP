'''
各クラスタのタグの頻度のヒストグラムとcsvを作成
'''
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
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
SAVE_DIR = f'{EXPT}/clustering/cluster_tag_freq/{TAG_PREPROCESS}/num_img_cluster={NUM_IMG_CLUSTERS}_num_tag_cluster={NUM_TAG_CLUSTERS}/{DATASET}'
os.makedirs(SAVE_DIR, exist_ok=True)

# パス，ラベル(クラスタID)の取得
_, tag_paths = utils.load_dataset_paths(DATASET)
tag_paths = np.asarray(tag_paths)
img_cluster_id = np.load(IMG_CLUSTER_PATH)['arr_0'].astype(np.int64)
tag_cluster_id = np.load(TAG_CLUSTER_PATH)['arr_0'].astype(np.int64)

# タグの一覧を取得(train全体の頻度順)
tag_list = utils.get_tag_list()

# クラスタ別のタグのリストを作成
# tags_list_cluster[i][j]: 印象クラスタi，画像クラスタjのフォントのタグのリスト
tags_list_cluster = [
    [
        [utils.get_font_tags(tag_path) for tag_path in tag_paths[(tag_cluster_id == i) & (img_cluster_id == j)]]
        for j in range(NUM_IMG_CLUSTERS)
    ]
    for i in range(NUM_TAG_CLUSTERS)
]

# 各クラスターのタグの頻度を計算
# frequency_list_cluster[i][j]: 印象クラスタi，画像クラスタjのフォントのタグの頻度(tag_list順)
frequency_list_cluster = []
for tag_cluster in tags_list_cluster:
    cluster_frequencies = []
    for img_cluster in tag_cluster:
        # 各画像クラスターのタグの頻度をカウント
        tag_counter = Counter(tag for tags in img_cluster for tag in tags)
        # tag_listの順序に従って値を取得
        cluster_frequencies.append([tag_counter[tag] for tag in tag_list])
    frequency_list_cluster.append(cluster_frequencies)

# 各クラスターのタグの頻度をプロット (印象クラスタ，画像クラスタ別)
for i in range(NUM_TAG_CLUSTERS):
    for j in range(NUM_IMG_CLUSTERS):
        fig, ax = plt.subplots(figsize=(24, 6))
        plt.bar(tag_list, frequency_list_cluster[i][j], edgecolor='black', width=1)
        plt.xticks(fontsize=5.3, rotation=90)
        plt.xlabel(f'Tag')
        plt.ylabel('Frequency')
        plt.savefig(f'{SAVE_DIR}/[img_cluster, tag_cluster]=[{i+1}, {j+1}].png', dpi=300, bbox_inches='tight')
        plt.close()

# 各クラスターのタグの頻度をプロット (画像クラスタ，印象クラスタ別)
for i in range(NUM_TAG_CLUSTERS):
    for j in range(NUM_IMG_CLUSTERS):
        fig, ax = plt.subplots(figsize=(24, 6))
        plt.bar(tag_list, frequency_list_cluster[i][j], edgecolor='black', width=1)
        plt.xticks(fontsize=5.3, rotation=90)
        plt.xlabel(f'Tag')
        plt.ylabel('Frequency')
        plt.savefig(f'{SAVE_DIR}/[img_cluster, tag_cluster]=[{i+1}, {j+1}].png', dpi=300, bbox_inches='tight')
        plt.close()
    
# 各クラスターのタグの頻度をプロット (印象クラスタ別)
for i in range(NUM_TAG_CLUSTERS):
    for j in range(NUM_IMG_CLUSTERS):
        fig, ax = plt.subplots(figsize=(24, 6))
        plt.bar(tag_list, frequency_list_cluster[i][j], edgecolor='black', width=1)
        plt.xticks(fontsize=5.3, rotation=90)
        plt.xlabel(f'Tag')
        plt.ylabel('Frequency')
        plt.savefig(f'{SAVE_DIR}/[img_cluster, tag_cluster]=[{i+1}, {j+1}].png', dpi=300, bbox_inches='tight')
        plt.close()

# 各クラスターのタグの頻度をプロット (画像クラスタ別)
for i in range(NUM_TAG_CLUSTERS):
    for j in range(NUM_IMG_CLUSTERS):
        fig, ax = plt.subplots(figsize=(24, 6))
        plt.bar(tag_list, frequency_list_cluster[i][j], edgecolor='black', width=1)
        plt.xticks(fontsize=5.3, rotation=90)
        plt.xlabel(f'Tag')
        plt.ylabel('Frequency')
        plt.savefig(f'{SAVE_DIR}/[img_cluster, tag_cluster]=[{i+1}, {j+1}].png', dpi=300, bbox_inches='tight')
        plt.close()



# DATASET全フォントの印象タグの頻度
fig, ax = plt.subplots(figsize=(24, 6))
tag_freq = Counter(tag for tag_path in tag_paths for tag in utils.get_font_tags(tag_path))
sorted_tag_freq = [tag_freq[tag] for tag in tag_list]
plt.bar(tag_list, sorted_tag_freq, edgecolor='black', width=1)
plt.xticks(fontsize=5.3, rotation=90)
plt.xlabel(f'Tag')
plt.ylabel('Frequency')
plt.savefig(f'{SAVE_DIR}/total.png', dpi=300, bbox_inches='tight')
plt.close()

# csvに保存
# frequency_list_to_write = copy.deepcopy(frequency_list)
# for i, sublist in enumerate(frequency_list_to_write):
#     sublist.insert(0, f'cluster{i}')
# first_row = [''] + tag_list
# last_row = ['total'] + list(tag_freq.values())
# frequency_list_to_write.insert(0, first_row)
# frequency_list_to_write.append(last_row)
# utils.save_list_to_csv(frequency_list_to_write, f'{SAVE_DIR}/tag_freq.csv')