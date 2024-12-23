'''
共埋め込み後の画像特徴と印象特徴をtSNEで可視化
指定したレイヤのクラスタで色分け
'''
import torch
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
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
IMG_CLUSTER_PATH = params.img_cluster_path
TAG_CLUSTER_PATH = params.tag_cluster_path
EMBEDDED_IMG_FEATURE_PATH = params.embedded_img_feature_path
EMBEDDED_TAG_FEATURE_PATH = params.embedded_tag_feature_path

# ディレクトリの作成
SAVE_DIR = f'{BASE_DIR}/tSNE/{DATASET}'
os.makedirs(f'{SAVE_DIR}', exist_ok=True)

# 学習データでtSNE
tSNE_filename = f'{BASE_DIR}/tSNE/tSNE_model.pkl'
if os.path.exists(tSNE_filename):
    with open(tSNE_filename, 'rb') as f:
        tSNE = pickle.load(f)
    print('Loaded existing tSNE model.')
else:
    print('tSNE start')
    embedded_img_feature = torch.load(f'{BASE_DIR}/feature/embedded_img_feature/train.pth')
    embedded_tag_feature = torch.load(f'{BASE_DIR}/feature/embedded_tag_feature/train.pth')
    feature = torch.concatenate([embedded_img_feature, embedded_tag_feature], dim=0)
    feature = feature.to('cpu').detach().numpy().copy()
    tSNE = TSNE(initialization='pca', metric='euclidean', n_jobs=-1, random_state=7, verbose=True).fit(feature)
    with open(tSNE_filename, 'wb') as f:
        pickle.dump(tSNE, f)
    print('tSNE end')
    print('Calculated and saved new tSNE.')

# tSENで埋め込み
tSNE_feature_filename = f'{SAVE_DIR}/tSNE_feature.npz'
if os.path.exists(tSNE_feature_filename):
    tSNE_embedding = np.load(tSNE_feature_filename)['arr_0']
    print('Loaded existing tSNE feature.')
else:
    print('tSNE embedding start')
    embedded_img_feature = torch.load(EMBEDDED_IMG_FEATURE_PATH)
    embedded_tag_feature = torch.load(EMBEDDED_TAG_FEATURE_PATH)
    feature = torch.concatenate([embedded_img_feature, embedded_tag_feature], dim=0)
    feature = feature.to('cpu').detach().numpy().copy()
    tSNE_embedding = tSNE.transform(feature)
    np.savez_compressed(tSNE_feature_filename, tSNE_embedding)
    print('tSNE embedding end')
    print('Calculated and saved new tSNE feature.')

# モダリティ(画像/印象)で色分けしてプロット
X = tSNE_embedding[:,0]
Y = tSNE_embedding[:,1]
t = int(len(tSNE_embedding)/2)
fig, ax = plt.subplots(figsize=(16, 12))
plt.scatter(X[:t], Y[:t], c='#377eb8', label='img', alpha=0.8, s=1)
plt.scatter(X[t:], Y[t:], c='#ff7f00', label='tag', alpha=0.8, s=1)
plt.legend()
plt.xlim(X.min(), X.max())
plt.ylim(Y.min(), Y.max())
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig(f'{SAVE_DIR}/tSNE.png', bbox_inches='tight', dpi=300)
plt.close()