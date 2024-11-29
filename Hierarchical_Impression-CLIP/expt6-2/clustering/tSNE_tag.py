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
    i = ind['ind'][0]
    pos = sc.get_offsets()[i]
    annot_img.xy = (pos[0]+10, pos[1]+6)
    annot_text.xy = (pos[0]+0.3, pos[1]-9)
    
    fontname = Path(img_paths[i]).stem
    img = np.load(img_paths[i])['arr_0'][0]
    tags = utils.get_font_tags(tag_paths[i])
    
    imagebox.set_data(img)
    annot_text.set_text(f'{fontname} {tags}')


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
TAG_CLUSTER_PATH = params.tag_cluster_path
NUM_TAG_CLUSTERS = params.num_tag_clusters
TAG_PREPROCESS = params.tag_preprocess

SAVE_DIR = f'{EXPT}/clustering/tSNE/tag/{TAG_PREPROCESS}/{DATASET}'
os.makedirs(SAVE_DIR, exist_ok=True)


# 学習データでtSNE
tag_feature_path_train = f'{EXPT}/feature/tag_feature/{TAG_PREPROCESS}/train.pth'
tag_feature_train = torch.load(tag_feature_path_train).to('cpu').detach().numpy()
tSNE_filename = f'{EXPT}/clustering/tSNE/tag/{TAG_PREPROCESS}/tSNE_model.pkl'
if os.path.exists(tSNE_filename):
    with open(tSNE_filename, 'rb') as f:
        tSNE = pickle.load(f)
    print('Loaded existing tSNE model.')
else:
    print('tSNE start')
    tSNE = TSNE(initialization='pca', metric='euclidean', n_jobs=-1, random_state=7).fit(tag_feature_train)
    with open(tSNE_filename, 'wb') as f:
        pickle.dump(tSNE, f)
    print('tSNE end')
    print('Calculated and saved new tSNE.')

# 印象特徴の取得 & tSNE embedding
tag_feature = torch.load(TAG_FEATURE_PATH).to('cpu').detach().numpy()
tSNE_feature_filename = f'{EXPT}/clustering/tSNE/tag/{TAG_PREPROCESS}/{DATASET}/tSNE_feature.pkl'
if os.path.exists(tSNE_feature_filename):
    with open(tSNE_feature_filename, 'rb') as f:
        embedding = pickle.load(f)
    print('Loaded existing tSNE feature.')
else:
    print('tSNE embedding start')
    embedding = tSNE.transform(tag_feature)
    with open(tSNE_feature_filename, 'wb') as f:
        pickle.dump(embedding, f)
    print('tSNE embedding end')
    print('Calculated and saved new tSNE feature.')
X = embedding[:,0]
Y = embedding[:,1]

# パス，ラベル(クラスタID)の取得
tag_cluster_id = np.load(TAG_CLUSTER_PATH)['arr_0'].astype(np.int64)
number_of_clusters = max(tag_cluster_id)+1


# プロット(マウスオーバーで画像と印象タグを表示)
fig, ax = plt.subplots()
patches = [mpatches.Patch(color=plt.cm.tab10(i), label=f'{i}') for i in range(number_of_clusters)]
sc = plt.scatter(X, Y, c=plt.cm.tab10(np.asarray(tag_cluster_id, dtype=np.int64)), 
                 alpha=0.8, edgecolors='w', linewidths=0.1, s=10)

img_paths, tag_paths = utils.load_dataset_paths(DATASET)
img = np.load(img_paths[0])['arr_0'][0]
imagebox = OffsetImage(img, zoom=0.7, cmap='gray')
imagebox.image.axes = ax
annot_img = AnnotationBbox(imagebox, xy=(0,0), xycoords='data', boxcoords='offset points', pad=0,
                           arrowprops=dict( arrowstyle='->', connectionstyle='arc3,rad=-0.3'))
annot_img.set_visible(False)
ax.add_artist(annot_img)
annot_text = ax.annotate('', xy=(0,0), xytext=(20,20),textcoords='offset points', bbox=dict(boxstyle='round', fc='w'))
annot_text.set_visible(False)

fig.canvas.mpl_connect('motion_notify_event', hover)
plt.legend(handles=patches)
fig.set_size_inches(6.4*1.5, 4.8*1.5)
plt.savefig(f'{SAVE_DIR}/{NUM_TAG_CLUSTERS}.png', bbox_inches='tight', dpi=300)
# plt.show()
plt.close()