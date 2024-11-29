import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
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
EXPT = params.expt
DATASET = params.dataset
IMG_FEATURE_PATH = params.img_feature_path
IMG_CLUSTER_PATH = params.img_cluster_path

SAVE_DIR = f'{EXPT}/clustering/PCA/img_plot_as_image'
os.makedirs(SAVE_DIR, exist_ok=True)


# 学習データでPCAしたものを読み込み
PCA_filename = f'{EXPT}/clustering/PCA/img/PCA_model.pkl'
with open(PCA_filename, 'rb') as f:
    pca = pickle.load(f)

# 画像特徴の取得 & PCA
img_features = torch.load(IMG_FEATURE_PATH).to("cpu").detach().numpy()
embedding = pca.transform(img_features)
X = embedding[:,0]
Y = embedding[:,1]

# 画像でプロット
fig, ax = plt.subplots()
img_paths, _ = utils.load_dataset_paths(DATASET)
plt.scatter(X, Y, alpha=0)
for i in range(len(img_paths)):
    img = np.load(img_paths[i])["arr_0"].astype(np.float32)[0]
    imagebox = OffsetImage(img, zoom=0.08, cmap='gray')
    ab = AnnotationBbox(imagebox, (X[i], Y[i]), frameon=False, pad=0)
    ax.add_artist(ab)
plt.savefig(f'{SAVE_DIR}/{DATASET}.png', bbox_inches='tight', dpi=500)