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
import pickle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches

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

# クラスタのラベルをどの階層まで見るか
# 4以上にすると, 色が足りない
DEPTH = 3

# ディレクトリの作成
SAVE_DIR = f'{BASE_DIR}/PCA_cluster'
os.makedirs(f'{SAVE_DIR}', exist_ok=True)

# ラベル(クラスタID)の取得
img_cluster_id = utils.load_hierarchical_clusterID(IMG_CLUSTER_PATH, DEPTH)
tag_cluster_id = utils.load_hierarchical_clusterID(TAG_CLUSTER_PATH, DEPTH)

# PCA特徴の読み込み
PCA_feature_filename = f'{BASE_DIR}/PCA/{DATASET}/PCA_feature.npz'
embedding = np.load(PCA_feature_filename)['arr_0']
X = embedding[:,0]
Y = embedding[:,1]
X_img = X[:int(len(X)/2)]
X_tag = X[int(len(X)/2):]
Y_img = Y[:int(len(Y)/2)]
Y_tag = Y[int(len(Y)/2):]

# 画像のクラスタで色分け
fig, ax = plt.subplots(figsize=(6.4*1.5, 4.8*1.5))
for i, label in enumerate(np.unique(img_cluster_id)):
    # ラベルごとにデータをフィルタリング
    mask = img_cluster_id==label
    plt.scatter(X_img[mask], Y_img[mask], label=f'{label}_img', color=plt.cm.tab20(i*2), 
                alpha=0.8, edgecolors='w', linewidths=0.1, s=5)
    plt.scatter(X_tag[mask], Y_tag[mask], label=f'{label}_tag', color=plt.cm.tab20(i*2+1), 
                alpha=0.8, edgecolors='w', linewidths=0.1, s=5)
plt.legend()
plt.savefig(f'{SAVE_DIR}/{DATASET}_img.png', bbox_inches='tight', dpi=300)
plt.close()

# 印象のクラスタで色分け
fig, ax = plt.subplots(figsize=(6.4*1.5, 4.8*1.5))
for i, label in enumerate(np.unique(tag_cluster_id)):
    # ラベルごとにデータをフィルタリング
    mask = tag_cluster_id==label
    plt.scatter(X_tag[mask], Y_tag[mask], label=f'{label}_img', color=plt.cm.tab20(i*2), 
                alpha=0.8, edgecolors='w', linewidths=0.1, s=5)
    plt.scatter(X_img[mask], Y_img[mask], label=f'{label}_tag', color=plt.cm.tab20(i*2+1), 
                alpha=0.8, edgecolors='w', linewidths=0.1, s=5)
plt.legend()
plt.savefig(f'{SAVE_DIR}/{DATASET}_tag.png', bbox_inches='tight', dpi=300)
plt.close()