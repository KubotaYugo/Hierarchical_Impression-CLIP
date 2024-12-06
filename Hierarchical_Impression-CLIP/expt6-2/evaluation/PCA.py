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


def update_annot(ind):
    i = ind['ind'][0]
    pos = sc.get_offsets()[i]
    index = i%len(img_paths)
    fontname = img_paths[index][len(f'dataset/MyFonts_preprocessed/font_numpy_Impression-CLIP/{DATASET}/'):-4]
    if i<len(img_paths):
        annot_img.xy = (pos[0]+0.05, pos[1]+0.01)
        annot_text.xy = (pos[0]+0.03, pos[1]-0.01)
        img = np.load(img_paths[index])['arr_0'][0]
        imagebox.set_data(img)
        annot_text.set_text(fontname)
    else:
        annot_text.xy = (pos[0]+0.03, pos[1]-0.01)
        tags = utils.get_font_tags(tag_paths[index])
        annot_text.set_text(f'{fontname} {tags}')

def hover(event):
    vis = annot_img.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            if ind['ind'][0] < len(img_paths):
                annot_img.set_visible(True)
                annot_text.set_visible(True)
            else:
                annot_text.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot_img.set_visible(False)
                annot_text.set_visible(False)
                fig.canvas.draw_idle()

# define constant
params = utils.get_parameters()
DATASET = params.dataset
BASE_DIR = params.base_dir
IMG_CLUSTER_PATH = params.img_cluster_path
TAG_CLUSTER_PATH = params.tag_cluster_path
EMBEDDED_IMG_FEATURE_PATH = params.embedded_img_feature_path
EMBEDDED_TAG_FEATURE_PATH = params.embedded_tag_feature_path

# ディレクトリの作成
SAVE_DIR = f'{BASE_DIR}/PCA/{DATASET}'
os.makedirs(f'{SAVE_DIR}', exist_ok=True)

# 学習データでPCA
PCA_filename = f'{BASE_DIR}/PCA/PCA_model.pkl'
if os.path.exists(PCA_filename):
    with open(PCA_filename, 'rb') as f:
        pca = pickle.load(f)
    print('Loaded existing PCA model.')
else:
    print('PCA_start')
    embedded_img_feature_train = torch.load(f'{BASE_DIR}/feature/embedded_img_feature/train.pth')
    embedded_tag_feature_train = torch.load(f'{BASE_DIR}/feature/embedded_tag_feature/train.pth')
    feature = torch.concatenate([embedded_img_feature_train, embedded_tag_feature_train], dim=0)
    feature = feature.to('cpu').detach().numpy().copy()
    pca = PCA(n_components=100)
    pca.fit(feature)
    with open(PCA_filename, 'wb') as f:
        pickle.dump(pca, f)
    print('PCA_end')
    print('Calculated and saved new PCA.')

# 特徴の取得 & 埋め込み
PCA_feature_filename = f'{BASE_DIR}/PCA/{DATASET}/PCA_feature.pkl'
if os.path.exists(PCA_feature_filename):
    with open(PCA_feature_filename, 'rb') as f:
        embedding = pickle.load(f)
    print('Loaded existing PCA feature.')
else:
    print('PCA embedding start')
    embedded_img_feature = torch.load(EMBEDDED_IMG_FEATURE_PATH)
    embedded_tag_feature = torch.load(EMBEDDED_TAG_FEATURE_PATH)
    feature = torch.concatenate([embedded_img_feature, embedded_tag_feature], dim=0)
    feature = feature.to('cpu').detach().numpy().copy()
    embedding = pca.transform(feature)
    with open(PCA_feature_filename, 'wb') as f:
        pickle.dump(embedding, f)
    print('PCA embedding end')
    print('Calculated and saved new PCA feature.')

# 主成分方向の分布
N = 10
df = pd.DataFrame(embedding[:, :N])
df = df.assign(modal='img')
t = int(len(embedding)/2)
df.loc[t:, 'modal'] = 'tag'
sns.pairplot(df, hue='modal', plot_kws={'s':10})
plt.savefig(f'{SAVE_DIR}/PCA.png', bbox_inches='tight', dpi=500)
plt.close()

# 累積寄与率
plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), '-o', markersize=3)
plt.plot([0] + list(pca.explained_variance_ratio_), '-o', markersize=3)
plt.xlim(0, 100)
plt.ylim(0, 1.0)
plt.xlabel('Number of principal components')
plt.ylabel('Cumulative contribution rate')
plt.grid()
plt.savefig(f'{SAVE_DIR}/contribution_rate.png', bbox_inches='tight', dpi=300)
plt.close()

# # 第1，第2主成分方向のプロット
X = embedding[:,0]
Y = embedding[:,1]
fig, ax = plt.subplots(figsize=(16, 12))
plt.scatter(X[:t], Y[:t], c='#377eb8', label='img', alpha=0.8, s=1)
plt.scatter(X[t:], Y[t:], c='#ff7f00', label='tag', alpha=0.8, s=1)
plt.legend()
plt.xlim(X.min(), X.max())
plt.ylim(Y.min(), Y.max())
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig(f'{SAVE_DIR}/PC1_PC2.png', bbox_inches='tight', dpi=300)
plt.close()

# ラベル(クラスタID)の取得
img_cluster_id = np.load(IMG_CLUSTER_PATH)['arr_0'].astype(np.int64)
tag_cluster_id = np.load(TAG_CLUSTER_PATH)['arr_0'].astype(np.int64)

# 各モダリティのクラスタ別に色分け
for ANNOTATE_WITH in ['img', 'tag']:
    # マウスオーバーで画像とクラス，ファイル名を表示
    fig, ax = plt.subplots(figsize=(6.4*1.5, 4.8*1.5))
    img_paths, tag_paths = utils.load_dataset_paths(DATASET)
    img = np.load(img_paths[0])['arr_0'][0]
    imagebox = OffsetImage(img, zoom=0.7, cmap='gray')
    imagebox.image.axes = ax

    if ANNOTATE_WITH=='img':
        labels = list(img_cluster_id*2)+list(img_cluster_id*2+1)
    elif ANNOTATE_WITH=='tag':
        labels = list(tag_cluster_id*2)+list(tag_cluster_id*2+1)
    modality = ['img', 'tag']
    patches = [mpatches.Patch(color=plt.cm.tab20(i), label=f'cluster{i//2}_{modality[i%2]}') for i in range(20)]
    sc = plt.scatter(X, Y, c=plt.cm.tab20(np.asarray(labels, dtype=np.int64)), 
                     alpha=0.8, edgecolors='w', linewidths=0.1, s=5)
    
    annot_img = AnnotationBbox(imagebox, xy=(0,0), xycoords='data', boxcoords='offset points', pad=0,
                            arrowprops=dict( arrowstyle='->', connectionstyle='arc3,rad=-0.3'))
    annot_img.set_visible(False)
    ax.add_artist(annot_img)

    annot_text = ax.annotate('', xy=(0,0), xytext=(20,20), textcoords='offset points', bbox=dict(boxstyle='round', fc='w'))
    annot_text.set_visible(False)

    fig.canvas.mpl_connect('motion_notify_event', hover)
    # plt.legend(handles=patches)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max()) 
    plt.savefig(f'{SAVE_DIR}/PC1_PC2_{ANNOTATE_WITH}.png', bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()