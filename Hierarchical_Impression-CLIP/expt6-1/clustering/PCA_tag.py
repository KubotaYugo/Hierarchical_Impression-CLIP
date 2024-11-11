import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
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
TAG_CLUSTER_PATH = params.tag_cluster_path
NUM_TAG_CLUSTERS = params.num_tag_clusters
TAG_PREPROCESS = params.tag_preprocess

SAVE_DIR = f'{EXPT}/clustering/PCA/tag/{DATASET}/{TAG_PREPROCESS}/{NUM_TAG_CLUSTERS}'
os.makedirs(SAVE_DIR, exist_ok=True)


# 学習データでPCA
tag_feature_path_train = f'{EXPT}/feature/tag_feature/train/{TAG_PREPROCESS}.pth'
tag_features_train = torch.load(tag_feature_path_train).to("cpu").detach().numpy()

PCA_filename = f'{SAVE_DIR}/PCA_model.pkl'
if os.path.exists(PCA_filename):
    with open(PCA_filename, 'rb') as f:
        pca = pickle.load(f)
    print("Loaded existing PCA model.")
else:
    print("PCA_start")
    pca = PCA(n_components=100)
    pca.fit(tag_features_train)
    with open(PCA_filename, 'wb') as f:
        pickle.dump(pca, f)
    print("PCA_uend")
    print("Calculated and saved new PCA.")

# 印象特徴の取得&PCA
tag_features = torch.load(TAG_FEATURE_PATH).to("cpu").detach().numpy()
embedding = pca.transform(tag_features)
X = embedding[:,0]
Y = embedding[:,1]

# パス，ラベル(クラスタID)の取得
tag_cluster_id = np.load(TAG_CLUSTER_PATH)["arr_0"].astype(np.int64)
number_of_clusters = max(tag_cluster_id)+1

# 第N主成分のうち2つの主成分を軸として特徴量分布を可視化
N = 10
df = pd.DataFrame(embedding[:, :N])
df['cluste_id'] = tag_cluster_id
sns.pairplot(df, hue="cluste_id", plot_kws={'s':2}, palette='tab10')
plt.savefig(f'{SAVE_DIR}/PCA.png', bbox_inches='tight', dpi=300)
plt.close()

# 累積寄与率
plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o", markersize=3)
plt.plot([0] + list(pca.explained_variance_ratio_), "-o", markersize=3)
plt.xlim(0, 100)
plt.ylim(0, 1.0)
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution rate")
plt.grid()
plt.savefig(f"{SAVE_DIR}/contribution_rate.png", bbox_inches='tight', dpi=300)
plt.close()

# プロット(マウスオーバーで画像と印象タグを表示)
fig, ax = plt.subplots()
patches = [mpatches.Patch(color=plt.cm.tab10(i), label=f'{i}') for i in range(number_of_clusters)]
sc = plt.scatter(X, Y, c=plt.cm.tab10(np.asarray(tag_cluster_id, dtype=np.int64)), 
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
plt.savefig(f'{SAVE_DIR}/PC1_PC2.png', bbox_inches='tight', dpi=300)
# plt.show()
plt.close()