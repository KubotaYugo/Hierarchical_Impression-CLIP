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
from lib import utils

# define constant
params = utils.get_parameters()
DATASET = params.dataset
BASE_DIR = params.base_dir
EMBEDDED_IMG_FEATURE_PATH = params.embedded_img_feature_path
EMBEDDED_TAG_FEATURE_PATH = params.embedded_tag_feature_path

# ディレクトリの作成
SAVE_DIR = f"{BASE_DIR}/PCA/{DATASET}"
os.makedirs(f"{SAVE_DIR}", exist_ok=True)

# PCAの準備 (学習データ)
embedded_img_feature = torch.load(f'{BASE_DIR}/feature/embedded_img_feature/train.pth')
embedded_tag_feature = torch.load(f'{BASE_DIR}/feature/embedded_tag_feature/train.pth')
feature = torch.concatenate([embedded_img_feature, embedded_tag_feature], dim=0)
feature = feature.to('cpu').detach().numpy().copy()
pca = PCA(n_components=100)
pca.fit(feature)

# 埋め込み
N = 5
embedded_img_feature = torch.load(EMBEDDED_IMG_FEATURE_PATH)
embedded_tag_feature = torch.load(EMBEDDED_TAG_FEATURE_PATH)
feature = torch.concatenate([embedded_img_feature, embedded_tag_feature], dim=0)
feature = feature.to('cpu').detach().numpy().copy()
embedding = pca.transform(feature)
df = pd.DataFrame(embedding[:, :N])
df = df.assign(modal="img")
t = len(embedded_img_feature)
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
plt.scatter(X[:t], Y[:t], c='#377eb8', label="img", alpha=0.8, s=1)
plt.scatter(X[t:], Y[t:], c='#ff7f00', label="tag", alpha=0.8, s=1)
plt.legend()
plt.xlim(X.min(), X.max())
plt.ylim(Y.min(), Y.max())
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig(f"{SAVE_DIR}/PC1_PC2.png", bbox_inches='tight', dpi=300)
plt.close()