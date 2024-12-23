'''
印象特徴のノルムの大きさ毎の分布(ヒストグラム)を見る
'''

import torch
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
TAG_FEATURE_PATH = params.tag_feature_path
TAG_PREPROCESS = params.tag_preprocess
NUMBER_OF_BINS = 50

# ディレクトリの作成
SAVE_DIR = f'{EXPT}/clustering/tag_feature_norm/{TAG_PREPROCESS}'
os.makedirs(SAVE_DIR, exist_ok=True)

# 印象特徴の取得 & ノルムの計算
tag_features = torch.load(TAG_FEATURE_PATH)
tag_feature_norms = torch.linalg.norm(tag_features, ord=2, dim=1)
tag_feature_norms = tag_feature_norms.to('cpu').detach().numpy()
x_min = tag_feature_norms.min()
x_max = tag_feature_norms.max()

# 印象タグの個数を計算
_, tag_paths = utils.load_dataset_paths(DATASET)
number_of_tags = [len(utils.get_font_tags(tag_path)) for tag_path in tag_paths]
number_of_tags = np.asarray(number_of_tags)

# ヒストグラムの作成 (まとめて色分けしてプロット)
stacked_hist = [tag_feature_norms[number_of_tags==i+1] for i in range(10)]
labels = [i+1 for i in range(10)]
fig, ax = plt.subplots(figsize=(6.4*1.5, 4.8*1.5))
plt.hist(stacked_hist, bins=NUMBER_OF_BINS, stacked=True, label=labels)
plt.xlim(x_min, x_max)
plt.xlabel('2-norm')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(f'{SAVE_DIR}/{DATASET}.png', bbox_inches='tight', dpi=300)
# plt.show()

# ヒストグラムの作成 (別々にプロット)
fig, ax = plt.subplots(10, 1, figsize=(4, 12), constrained_layout=True)
for i in range(10):
    ax[i].hist(stacked_hist[i], bins=NUMBER_OF_BINS)
    ax[i].set_xlabel(f'Number of Tags: {i+1}')
    ax[i].set_ylabel('Frequency')
    ax[i].set_xlim(x_min, x_max)
plt.savefig(f'{SAVE_DIR}/{DATASET}_separete.png', bbox_inches='tight', dpi=300)
plt.close()