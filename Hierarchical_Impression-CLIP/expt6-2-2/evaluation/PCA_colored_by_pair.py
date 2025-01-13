'''
共埋め込み後の画像特徴と印象特徴をPCAで可視化
画像特徴: 自身のy座標で色付け
印象特徴: ペアの画像特徴y座標で色付け
'''
import matplotlib.pyplot as plt
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
    plt.savefig(f'{SAVE_DIR}/{name}.png', bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()


# define constant
params = utils.get_parameters()
DATASET = params.dataset
BASE_DIR = params.base_dir
EMBEDDED_IMG_FEATURE_PATH = params.embedded_img_feature_path
EMBEDDED_TAG_FEATURE_PATH = params.embedded_tag_feature_path

# ディレクトリの作成
SAVE_DIR = f'{BASE_DIR}/PCA_colored_by_pair/{DATASET}'
os.makedirs(f'{SAVE_DIR}', exist_ok=True)

# PCA特徴の読み込み
PCA_feature_filename = f'{BASE_DIR}/PCA/{DATASET}/PCA_feature.npz'
embedding = np.load(PCA_feature_filename)['arr_0']
X = embedding[:,0]
Y = embedding[:,1]
X_img = X[:int(len(X)/2)]
X_tag = X[int(len(X)/2):]
Y_img = Y[:int(len(Y)/2)]
Y_tag = Y[int(len(Y)/2):]

# 画像特徴のx座標で色付け
plot(X, Y, np.tile(X_img, 2), f'img_x')
plot(X_img, Y_img, X_img, f'img_x_img')
plot(X_tag, Y_tag, X_img, f'img_x_tag')

# 画像特徴のy座標で色付け
plot(X, Y, np.tile(Y_img, 2), f'img_y')
plot(X_img, Y_img, Y_img, f'img_y_img')
plot(X_tag, Y_tag, Y_img, f'img_y_tag')

# 印象特徴のx座標で色付け
plot(X, Y, np.tile(Y_tag, 2), f'tag_x')
plot(X_img, Y_img, X_tag, f'tag_x_img')
plot(X_tag, Y_tag, X_tag, f'tag_x_tag')

# 印象特徴のy座標で色付け
plot(X, Y, np.tile(Y_tag, 2), f'tag_y')
plot(X_img, Y_img, Y_tag, f'tag_y_img')
plot(X_tag, Y_tag, Y_tag, f'tag_y_tag')