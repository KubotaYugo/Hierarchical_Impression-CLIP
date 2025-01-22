import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
import pickle

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


def update_annot(ind):
    i = ind["ind"][0]
    pos = sc.get_offsets()[i]
    annot_img.xy = (pos[0]+0.3, pos[1]+0.1)
    annot_text.xy = (pos[0]+0.3, pos[1]-0.1)
    
    fontname = Path(img_paths[i]).stem
    img = np.load(img_paths[i])["arr_0"][0]
    tags = utils.get_font_tags(tag_paths[i])
    
    imagebox.set_data(img)
    annot_text.set_text(f"{fontname} {tags}")

def hover(event):
    vis = annot_img.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot_img.set_visible(True)
            annot_text.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot_img.set_visible(False)
                annot_text.set_visible(False)
                fig.canvas.draw_idle()


# define constant
params = utils.get_parameters()
EXPT = params.expt
DATASET = params.dataset
TAG_FEATURE_PATH = params.tag_feature_path
TAG_PREPROCESS = params.tag_preprocess

SAVE_DIR = f'{EXPT}/clustering/PCA/tag_number_of_tags_normalized/{TAG_PREPROCESS}'
os.makedirs(SAVE_DIR, exist_ok=True)

# 学習データでPCAしたものを読み込み
PCA_filename = f'{EXPT}/clustering/PCA/tag/{TAG_PREPROCESS}/PCA_model.pkl'
with open(PCA_filename, 'rb') as f:
    pca = pickle.load(f)

# 印象特徴の取得 & PCA
tag_features = torch.load(TAG_FEATURE_PATH).to("cpu").detach().numpy()
tag_features = tag_features / np.linalg.norm(tag_features, axis=1, keepdims=True)
embedding = pca.transform(tag_features)
X = embedding[:,0]
Y = embedding[:,1]

# タグの個数を取得
_, tag_paths = utils.load_dataset_paths(DATASET)
number_of_tags = [len(utils.get_font_tags(tag_path)) for tag_path in tag_paths]
number_of_tags = np.asarray(number_of_tags)-1

# プロット(マウスオーバーで画像と印象タグを表示)
fig, ax = plt.subplots()
patches = [mpatches.Patch(color=plt.cm.tab10(i), label=f'{i+1}') for i in range(max(number_of_tags)+1)]
sc = plt.scatter(X, Y, c=plt.cm.tab10(np.asarray(number_of_tags, dtype=np.int64)), 
                 alpha=0.8, edgecolors='w', linewidths=0.1, s=5)

img_paths, tag_paths = utils.load_dataset_paths(DATASET)
img = np.load(img_paths[0])["arr_0"][0]
imagebox = OffsetImage(img, zoom=0.7, cmap='gray')
imagebox.image.axes = ax
annot_img = AnnotationBbox(imagebox, xy=(0,0), xycoords="data", boxcoords="offset points", pad=0,
                           arrowprops=dict( arrowstyle="->", connectionstyle="arc3,rad=-0.3"))
annot_img.set_visible(False)
ax.add_artist(annot_img)
annot_text = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points", bbox=dict(boxstyle="round", fc="w"))
annot_text.set_visible(False)

fig.canvas.mpl_connect("motion_notify_event", hover)
plt.legend(handles=patches)
fig.set_size_inches(6.4*1.5, 4.8*1.5)
plt.savefig(f'{SAVE_DIR}/{DATASET}.png', bbox_inches='tight', dpi=300)
# plt.show()
plt.close()