'''
共埋め込み後の画像特徴と印象特徴の可視化
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
EMBEDDED_IMG_FEATURE_PATH = params.embedded_img_feature_path
EMBEDDED_TAG_FEATURE_PATH = params.embedded_tag_feature_path

# ディレクトリの作成
SAVE_DIR = f'{BASE_DIR}/PCA/{DATASET}'
os.makedirs(f'{SAVE_DIR}', exist_ok=True)

# 学習データでPCA
PCA_filename = f'{BASE_DIR}/PCA/PCA_model.pkl'
if os.path.exists(PCA_filename):
    with open(PCA_filename, 'rb') as f:
        pca = pickle.load(f)
    print('Loaded existing PCA model.')
else:
    print('PCA_start')
    embedded_img_feature_train = torch.load(f'{BASE_DIR}/feature/embedded_img_feature/train.pth')
    embedded_tag_feature_train = torch.load(f'{BASE_DIR}/feature/embedded_tag_feature/train.pth')
    feature = torch.concatenate([embedded_img_feature_train, embedded_tag_feature_train], dim=0)
    feature = feature.to('cpu').detach().numpy().copy()
    pca = PCA(n_components=100)
    pca.fit(feature)
    with open(PCA_filename, 'wb') as f:
        pickle.dump(pca, f)
    print('PCA_end')
    print('Calculated and saved new PCA.')

# 特徴の取得 & 埋め込み
PCA_feature_filename = f'{BASE_DIR}/PCA/{DATASET}/PCA_feature.npz'
if os.path.exists(PCA_feature_filename):
    embedding = np.load(PCA_feature_filename)['arr_0']
    print('Loaded existing PCA feature.')
else:
    print('PCA embedding start')
    embedded_img_feature = torch.load(EMBEDDED_IMG_FEATURE_PATH)
    embedded_tag_feature = torch.load(EMBEDDED_TAG_FEATURE_PATH)
    feature = torch.concatenate([embedded_img_feature, embedded_tag_feature], dim=0)
    feature = feature.to('cpu').detach().numpy().copy()
    embedding = pca.transform(feature)
    np.savez_compressed(PCA_feature_filename, embedding)
    print('PCA embedding end')
    print('Calculated and saved new PCA feature.')

DEPTH = 3
img_cluster_id = utils.load_hierarchical_clusterID(IMG_CLUSTER_PATH, DEPTH)

# 主成分方向の分布
N = 5
df = pd.DataFrame(embedding[:int(len(embedding)/2), :N])
df['cluster'] = img_cluster_id
sns.pairplot(df, hue='cluster', palette="tab10", plot_kws={'s':2})
plt.savefig(f'zzz.png', bbox_inches='tight', dpi=500)
plt.close()