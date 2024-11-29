'''
共埋め込み後の印象特徴の分布をタグの個数で色分けしてtSNEで可視化
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from openTSNE import TSNE
import pickle

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


# define constant
params = utils.get_parameters()
DATASET = params.dataset
BASE_DIR = params.base_dir
EMBEDDED_TAG_FEATURE_PATH = params.embedded_tag_feature_path

# 保存用フォルダの準備
DATASET = 'train'
SAVE_DIR = f"{BASE_DIR}/tSNE_tag_embedding/{DATASET}"
os.makedirs(f"{SAVE_DIR}", exist_ok=True)

# 特徴量の読み込み (train)
embedded_tag_feature_path_train = f'{BASE_DIR}/feature/embedded_tag_feature/train.pth'
embedded_tag_feature = torch.load(embedded_tag_feature_path_train)
embedded_tag_feature = embedded_tag_feature.to('cpu').detach().numpy().copy()

# 学習データでtSNE
tSNE_filename = f'{BASE_DIR}/tSNE_tag_embedding/tSNE_model.pkl'
if os.path.exists(tSNE_filename):
    with open(tSNE_filename, 'rb') as f:
        tSNE = pickle.load(f)
    print('Loaded existing tSNE model.')
else:
    print('tSNE start')
    tSNE = TSNE(initialization="pca", metric="euclidean", 
                n_jobs=-1, random_state=7, verbose=True).fit(embedded_tag_feature)
    with open(tSNE_filename, 'wb') as f:
        pickle.dump(tSNE, f)
    print('tSNE end')
    print('Calculated and saved new tSNE.')

# タグの個数を取得
_, tag_paths = utils.load_dataset_paths(DATASET)
number_of_tags = [len(utils.get_font_tags(tag_path)) for tag_path in tag_paths]
number_of_tags = np.asarray(number_of_tags)

# 特徴量の読み込み (DATASET)
embedded_tag_feature = torch.load(EMBEDDED_TAG_FEATURE_PATH)
embedded_tag_feature = embedded_tag_feature.to('cpu').detach().numpy().copy()

# tSENで埋め込み
tSNE_feature_filename = f'{SAVE_DIR}/tSNE_feature.npz'
if os.path.exists(tSNE_feature_filename):
    tSNE_embedding = np.load(tSNE_feature_filename)['arr_0']
    print('Loaded existing tSNE feature.')
else:
    print('tSNE embedding start')
    tSNE_embedding = tSNE.transform(embedded_tag_feature)
    np.savez_compressed(tSNE_feature_filename, tSNE_embedding)
    print('tSNE embedding end')
    print('Calculated and saved new tSNE feature.')
X = tSNE_embedding[:, 0]
Y = tSNE_embedding[:, 1]

# タグの個数で色分けして印象特徴をプロット
fig, ax = plt.subplots(figsize=(6.4*1.5, 4.8*1.5))
for i in range(1, 10+1):
    plt.scatter(X[number_of_tags==i], Y[number_of_tags==i], 
                alpha=0.8, edgecolors='w', linewidths=0.1, s=5, label=f'{i}')
plt.legend()
plt.savefig(f"{SAVE_DIR}/tSNE.png", bbox_inches='tight', dpi=300)
plt.close()