'''
共埋め込み前後の印象特徴の分布をタグの個数で色分けしてtSNEで可視化
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from openTSNE import TSNE

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils



# define constant
BASE_DIR = utils.BASE_DIR

# 保存用フォルダの準備
DATASET = 'train'
SAVE_DIR = f"{BASE_DIR}/tSNE_tag_embedding/{DATASET}"
os.makedirs(f"{SAVE_DIR}", exist_ok=True)

# 特徴量の読み込み (train)
load_dir = f'{BASE_DIR}/features/{DATASET}'
tag_features_before = torch.load(f'{load_dir}/tag_features.pth')
tag_features_after = torch.load(f'{load_dir}/embedded_tag_features.pth')
tag_features_before = tag_features_before.to('cpu').detach().numpy().copy()
tag_features_after = tag_features_after.to('cpu').detach().numpy().copy()
tag_features = {'before':tag_features_before, 'after':tag_features_after}

# タグの個数を取得
_, tag_paths = utils.load_dataset_paths(DATASET)
number_of_tags = [len(utils.get_font_tags(tag_path)) for tag_path in tag_paths]
number_of_tags = np.asarray(number_of_tags)

for embed in ['before', 'after']:
    
    tag_feature = tag_features[embed]

    # tSENで埋め込み
    tSNE_feature_filename = f'{SAVE_DIR}/tSNE_feature_{embed}_embbeding.npz'
    if os.path.exists(tSNE_feature_filename):
        tSNE_embedding = np.load(tSNE_feature_filename)['arr_0']
        print("Loaded existing t-SNE feature.")
    else:
        print(f"tSNE transform start ({DATASET})")
        tSNE_embedding = TSNE(initialization="pca", metric="euclidean", n_jobs=20, random_state=7, verbose=True).fit(tag_feature)
        np.savez_compressed(tSNE_feature_filename, tSNE_embedding)
        print(f"tSNE transform end ({DATASET})")
        print("Calculated and saved new t-SNE.")
    X = tSNE_embedding[:, 0]
    Y = tSNE_embedding[:, 1]

    # タグの個数で色分けして印象特徴をプロット
    fig, ax = plt.subplots(figsize=(6.4*1.5, 4.8*1.5))
    for i in range(1, 10+1):
        plt.scatter(X[number_of_tags==i], Y[number_of_tags==i], alpha=0.8, edgecolors='w', linewidths=0.1, s=5, label=f'{i}')
    plt.legend()
    plt.savefig(f"{SAVE_DIR}/tSNE_feature_{embed}_embbeding.png", bbox_inches='tight', dpi=500)
    plt.close()