'''
共埋め込み後の画像特徴と印象特徴をPCAで可視化
画像特徴: 自身のy座標で色付け
印象特徴: ペアの画像特徴y座標で色付け
'''
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


def plot(x, y, color, name):
    sort_indices = np.argsort(np.abs(color-np.mean(color)))
    x = x[sort_indices]
    y = y[sort_indices]
    color = np.asarray(color)
    color = color[sort_indices]
    scatter = plt.scatter(x, y, c=color, alpha=0.8, edgecolors='w', linewidths=0.1, s=1, cmap='jet')
    plt.colorbar(scatter)
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
    plt.savefig(f"{SAVE_DIR}/{name}.png", bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()


# define constant
params = utils.get_parameters()
DATASET = params.dataset
BASE_DIR = params.base_dir
EMBEDDED_IMG_FEATURE_PATH = params.embedded_img_feature_path
EMBEDDED_TAG_FEATURE_PATH = params.embedded_tag_feature_path

# ディレクトリの作成
SAVE_DIR = f"{BASE_DIR}/PCA_colored_by_pair"
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

# 第1，第2主成分方向のプロット (画像特徴のy座標で色付け)
X = embedding[:,0]
Y = embedding[:,1]
image_x = X[:len(embedded_img_feature)]
image_y = Y[:len(embedded_img_feature)]
tag_x = X[len(embedded_img_feature):]
tag_y = Y[len(embedded_img_feature):]
plot(X, Y, np.tile(image_y, 2), f'{DATASET}')
plot(image_x, image_y, image_y, f'{DATASET}_img')
plot(tag_x, tag_y, image_y, f'{DATASET}_tag')
