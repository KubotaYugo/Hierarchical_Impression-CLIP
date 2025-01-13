'''
共埋め込み後の画像特徴と印象特徴の可視化
(クラスタの学習を入れなかった場合)
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
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
IMG_CLUSTER_PATH = params.img_cluster_path
TAG_CLUSTER_PATH = params.tag_cluster_path

# ベースラインのハイパラ
BASE_DIR_BASELINE = params.base_dir_baseline
EMBEDDED_IMG_FEATURE_PATH_BASELINE = params.embedded_img_feature_path_baseline
EMBEDDED_TAG_FEATURE_PATH_BASELINE = params.embedded_tag_feature_path_baseline

# ディレクトリの作成
SAVE_DIR = f'{BASE_DIR}/PCA_baseline/{DATASET}'
os.makedirs(f'{SAVE_DIR}', exist_ok=True)

# 学習データでPCA
PCA_filename = f'{BASE_DIR}/PCA_baseline/PCA_model.pkl'
if os.path.exists(PCA_filename):
    with open(PCA_filename, 'rb') as f:
        pca = pickle.load(f)
    print('Loaded existing PCA model.')
else:
    print('PCA_start')
    embedded_img_feature_train = torch.load(f'{BASE_DIR_BASELINE}/feature/embedded_img_feature/train.pth')
    embedded_tag_feature_train = torch.load(f'{BASE_DIR_BASELINE}/feature/embedded_tag_feature/train.pth')
    feature = torch.concatenate([embedded_img_feature_train, embedded_tag_feature_train], dim=0)
    feature = feature.to('cpu').detach().numpy().copy()
    pca = PCA(n_components=100)
    pca.fit(feature)
    with open(PCA_filename, 'wb') as f:
        pickle.dump(pca, f)
    print('PCA_end')
    print('Calculated and saved new PCA.')

# 特徴の取得 & 埋め込み
PCA_feature_filename = f'{BASE_DIR}/PCA_baseline/{DATASET}/PCA_feature.npz'
if os.path.exists(PCA_feature_filename):
    embedding = np.load(PCA_feature_filename)['arr_0']
    print('Loaded existing PCA feature.')
else:
    print('PCA embedding start')
    embedded_img_feature = torch.load(EMBEDDED_IMG_FEATURE_PATH_BASELINE)
    embedded_tag_feature = torch.load(EMBEDDED_TAG_FEATURE_PATH_BASELINE)
    feature = torch.concatenate([embedded_img_feature, embedded_tag_feature], dim=0)
    feature = feature.to('cpu').detach().numpy().copy()
    embedding = pca.transform(feature)
    np.savez_compressed(PCA_feature_filename, embedding)
    print('PCA embedding end')
    print('Calculated and saved new PCA feature.')

# 主成分方向の分布
N = 5
df = pd.DataFrame(embedding[:, :N])
df = df.assign(modal='img')
t = int(len(embedding)/2)
df.loc[t:, 'modal'] = 'tag'
sns.pairplot(df, hue='modal', plot_kws={'s':10})
plt.savefig(f'{SAVE_DIR}/PCA.png', bbox_inches='tight', dpi=500)
plt.close()

# 累積寄与率
plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), '-o', markersize=3)
plt.plot([0] + list(pca.explained_variance_ratio_), '-o', markersize=3)
plt.xlim(0, 100)
plt.ylim(0, 1.0)
plt.xlabel('Number of principal components')
plt.ylabel('Cumulative contribution rate')
plt.grid()
plt.savefig(f'{SAVE_DIR}/contribution_rate.png', bbox_inches='tight', dpi=300)
plt.close()

# # 第1，第2主成分方向のプロット
X = embedding[:,0]
Y = embedding[:,1]
fig, ax = plt.subplots(figsize=(16, 12))
plt.scatter(X[:t], Y[:t], c='#377eb8', label='img', alpha=0.8, s=1)
plt.scatter(X[t:], Y[t:], c='#ff7f00', label='tag', alpha=0.8, s=1)
plt.legend()
plt.xlim(X.min(), X.max())
plt.ylim(Y.min(), Y.max())
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig(f'{SAVE_DIR}/PC1_PC2.png', bbox_inches='tight', dpi=300)
plt.close()