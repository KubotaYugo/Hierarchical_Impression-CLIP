'''
共埋め込み後の画像特徴と印象特徴の可視化
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils

def plot_guide(ax):
    # 球体のガイドラインを描画
    guide_u = np.linspace(0, 2 * np.pi, 100)
    guide_v = np.linspace(0, np.pi, 100)
    guide_x = np.outer(np.sin(guide_v), np.cos(guide_u))
    guide_y = np.outer(np.sin(guide_v), np.sin(guide_u))
    guide_z = np.outer(np.cos(guide_v), np.ones_like(guide_u))
    ax.plot_wireframe(guide_x, guide_y, guide_z, color="gray", alpha=0.1)
    return ax


# define constant
params = utils.get_parameters()
DATASET = params.dataset
BASE_DIR = params.base_dir
IMG_CLUSTER_PATH = params.img_cluster_path
TAG_CLUSTER_PATH = params.tag_cluster_path
EMBEDDED_IMG_FEATURE_PATH = params.embedded_img_feature_path
EMBEDDED_TAG_FEATURE_PATH = params.embedded_tag_feature_path

# PCA特徴の取得 & L2norm=1になるように正規化
PCA_feature_filename = f'{BASE_DIR}/PCA/{DATASET}/PCA_feature.pkl'
with open(PCA_feature_filename, 'rb') as f:
    embedding = pickle.load(f)
embedding = embedding[:,:3]
row_norms = np.linalg.norm(embedding, ord=2, axis=1, keepdims=True)
normalized_embedding = embedding / row_norms

# プロット用の座標を取得
x = normalized_embedding[:,0]
y = normalized_embedding[:,1]
z = normalized_embedding[:,2]
t = int(len(x)/2)
img_x = x[:t]
img_y = y[:t]
img_z = z[:t]
tag_x = x[t:]
tag_y = y[t:]
tag_z = z[t:]

# モダリティ別に色分け
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax = plot_guide(ax)
ax.scatter(img_x, img_y, img_z, label='img', s=1)
ax.scatter(tag_x, tag_y, tag_z, label='tag', s=1)
ax.set_box_aspect([1, 1, 1])  # 等方性に
ax.legend()
plt.show()

# 画像特徴のz座標で同じフォントの画像/印象を色付け
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax = plot_guide(ax)
scatter = ax.scatter(x, y, z, c=list(img_z)*2, alpha=0.8, edgecolors='w', linewidths=0.1, s=5, cmap='jet')
plt.show()
plt.close()

# 画像/印象それぞれのクラスタで色分け
for ANNOTATE_WITH in ['img', 'tag']:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax = plot_guide(ax)
    img_cluster_id = np.load(IMG_CLUSTER_PATH)["arr_0"].astype(np.int64)
    tag_cluster_id = np.load(TAG_CLUSTER_PATH)["arr_0"].astype(np.int64)
    if ANNOTATE_WITH=='img':
        labels = list(img_cluster_id*2)+list(img_cluster_id*2+1)
    elif ANNOTATE_WITH=='tag':
        labels = list(tag_cluster_id*2)+list(tag_cluster_id*2+1)
    modality = ['img', 'tag']
    patches = [mpatches.Patch(color=plt.cm.tab20(i), label=f"cluster{i//2}_{modality[i%2]}") for i in range(20)]
    ax.scatter(x, y, z, c=plt.cm.tab20(np.asarray(labels, dtype=np.int64)), 
               s=5, alpha=0.8, edgecolors='w', linewidths=0.1)
    plt.show()
    plt.close()