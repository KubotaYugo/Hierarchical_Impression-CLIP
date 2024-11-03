'''
共埋め込み後の画像特徴と印象特徴の可視化
'''
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils

# define constant
EXP = utils.EXP
IMG_HIERARCHY_PATH = utils.IMG_HIERARCHY_PATH
TAG_HIERARCHY_PATH = utils.TAG_HIERARCHY_PATH

# PCAの準備 (学習データ)
load_dir = f'{EXP}/features/train'
embedded_img_features = torch.load(f'{load_dir}/embedded_img_features.pth')
embedded_tag_features = torch.load(f'{load_dir}/embedded_tag_features.pth')
features = torch.concatenate([embedded_img_features, embedded_tag_features], dim=0)
features = features.to('cpu').detach().numpy().copy()
pca = PCA(n_components=100)
pca.fit(features)

for DATASET in ['train', 'val', 'test']:
    # 保存用ディレクトリの作成
    SAVE_DIR = f"{EXP}/PCA/{DATASET}"
    os.makedirs(f"{SAVE_DIR}", exist_ok=True)

    # 埋め込み
    N = 5
    load_dir = f'{EXP}/features/{DATASET}'
    embedded_img_features = torch.load(f'{load_dir}/embedded_img_features.pth')
    embedded_tag_features = torch.load(f'{load_dir}/embedded_tag_features.pth')
    features = torch.concatenate([embedded_img_features, embedded_tag_features], dim=0)
    features = features.to('cpu').detach().numpy().copy()
    embedding = pca.transform(features)
    df = pd.DataFrame(embedding[:, :N])
    df = df.assign(modal="img")
    t = len(embedded_img_features)
    df.loc[t:, "modal"] = "tag"

    # 主成分方向の分布
    sns.pairplot(df, hue="modal", plot_kws={'s':10})
    plt.savefig(f"{SAVE_DIR}/PCA.png", bbox_inches='tight', dpi=500)
    plt.close()

    # 累積寄与率
    plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o", markersize=3)
    plt.plot([0] + list(pca.explained_variance_ratio_), "-o", markersize=3)
    plt.xlim(0, 100)
    plt.ylim(0, 1.0)
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative contribution rate")
    plt.grid()
    plt.savefig(f"{SAVE_DIR}/contribution_rate.png", bbox_inches='tight', dpi=300)
    plt.close()

    # 第1，第2主成分方向のプロット
    X = embedding[:,0]
    Y = embedding[:,1]
    fig, ax = plt.subplots(figsize=(16, 12))
    plt.scatter(X[:t], Y[:t], c='#377eb8', label="img", alpha=0.8, s=5)
    plt.scatter(X[t:], Y[t:], c='#ff7f00', label="tag", alpha=0.8, s=5)
    plt.legend()
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(f"{SAVE_DIR}/PC1_PC2.png", bbox_inches='tight', dpi=500)
    plt.close()