import torch
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from openTSNE import TSNE
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
    annot_img.xy = (pos[0]+10, pos[1]+6)
    annot_text.xy = (pos[0]+0.3, pos[1]-9)
    
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
EXPT = params['expt']
DATASET = params['dataset']
IMG_FEATURE_PATH = params['img_feature_path']
IMG_CLUSTER_PATH = params['img_cluster_path']
NUM_IMG_CLUSTERS = params['num_img_clusters']

SAVE_DIR = f'{EXPT}/clustering/tSNE_img/{DATASET}'
os.makedirs(SAVE_DIR, exist_ok=True)


# 学習データでtSNE
img_feature_path_train = f'{EXPT}/img_features/train.pth'
img_features_train = torch.load(img_feature_path_train).to("cpu").detach().numpy()
tSNE_filename = f'{SAVE_DIR}/image_cluster_tSNE_model.pkl'
if os.path.exists(tSNE_filename):
    with open(tSNE_filename, 'rb') as f:
        tSNE = pickle.load(f)
    print("Loaded existing t-SNE model.")
else:
    print("tSNE_start")
    tSNE = TSNE(initialization="pca", metric="euclidean", n_jobs=20, random_state=7).fit(img_features_train)
    with open(tSNE_filename, 'wb') as f:
        pickle.dump(tSNE, f)
    print("tSNE_end")
    print("Calculated and saved new t-SNE.")

# 画像特徴の取得&tSNE
img_features = torch.load(IMG_FEATURE_PATH).to("cpu").detach().numpy()
embedding = tSNE.transform(img_features)
X = embedding[:,0]
Y = embedding[:,1]

# パス，ラベル(クラスタID)の取得
img_cluster_id = np.load(IMG_CLUSTER_PATH)["arr_0"].astype(np.int64)
number_of_clusters = max(img_cluster_id)+1

# プロット(マウスオーバーで画像と印象タグを表示)
fig, ax = plt.subplots()
patches = [mpatches.Patch(color=plt.cm.tab10(i), label=f'{i}') for i in range(number_of_clusters)]
sc = plt.scatter(X, Y, c=plt.cm.tab10(np.asarray(img_cluster_id, dtype=np.int64)), 
                 alpha=0.8, edgecolors='w', linewidths=0.1, s=10)

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
plt.savefig(f'{SAVE_DIR}/{NUM_IMG_CLUSTERS}.png', bbox_inches='tight', dpi=300)
# plt.show()
plt.close()