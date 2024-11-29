import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
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

SAVE_DIR = f'{EXPT}/clustering/tSNE/img_plot_as_image'
os.makedirs(SAVE_DIR, exist_ok=True)

# tSNE embeddingの読み込み
tSNE_feature_filename = f'{EXPT}/clustering/tSNE/img/{DATASET}/tSNE_feature.pkl'
with open(tSNE_feature_filename, 'rb') as f:
    embedding = pickle.load(f)
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