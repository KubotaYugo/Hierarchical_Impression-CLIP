'''
共埋め込み後の画像特徴と印象特徴の可視化
'''
import torch
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from openTSNE import TSNE
import pickle

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


def update_annot(ind):
    i = ind["ind"][0]
    pos = sc.get_offsets()[i]
    index = i%len(img_paths)
    fontname = img_paths[index][len(f"dataset/MyFonts_preprocessed/font_numpy_Impression-CLIP/{DATASET}/"):-4]
    if i<len(img_paths):
        annot_img.xy = (pos[0]+0.5, pos[1]+0.5)
        annot_text.xy = (pos[0]+0.3, pos[1]-0.3)
        img = np.load(img_paths[index])["arr_0"][0]
        imagebox.set_data(img)
        annot_text.set_text(fontname)
    else:
        annot_text.xy = (pos[0]+0.3, pos[1]-0.3)
        tags = utils.get_font_tags(tag_paths[index])
        annot_text.set_text(f"{fontname} {tags}")

def hover(event):
    vis = annot_img.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            if ind["ind"][0] < len(img_paths):
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
IMG_CLUSTER_PATH = utils.IMG_CLUSTER_PATH
TAG_CLUSTER_PATH = utils.TAG_CLUSTER_PATH
BASE_DIR = utils.BASE_DIR

# 保存用フォルダの準備
DATASET = 'train'
SAVE_DIR = f"{BASE_DIR}/tSNE/{DATASET}"
os.makedirs(f"{SAVE_DIR}", exist_ok=True)

# 特徴量の読み込み (train)
load_dir = f'{BASE_DIR}/features/{DATASET}'
embedded_img_features = torch.load(f'{load_dir}/embedded_img_features.pth')
embedded_tag_features = torch.load(f'{load_dir}/embedded_tag_features.pth')
features = torch.concatenate([embedded_img_features, embedded_tag_features], dim=0)
features = features.to('cpu').detach().numpy().copy()

# tSNEの準備
tSNE_filename = f'{SAVE_DIR}/tSNE_model.pkl'
if os.path.exists(tSNE_filename):
    with open(tSNE_filename, 'rb') as f:
        tSNE = pickle.load(f)
    print("Loaded existing t-SNE model.")
else:
    print("tSNE_start")
    tSNE = TSNE(initialization="pca", metric="euclidean", n_jobs=20, random_state=7, verbose=True).fit(features)
    with open(tSNE_filename, 'wb') as f:
        pickle.dump(tSNE, f)
    print("tSNE_end")
    print("Calculated and saved new t-SNE.")


for DATASET in ['train', 'val', 'test']:
    # 保存用フォルダの準備
    SAVE_DIR = f"{BASE_DIR}/tSNE/{DATASET}"
    os.makedirs(f"{SAVE_DIR}", exist_ok=True)

    # 特徴量の読み込み
    load_dir = f'{BASE_DIR}/features/{DATASET}'
    embedded_img_features = torch.load(f'{load_dir}/embedded_img_features.pth')
    embedded_tag_features = torch.load(f'{load_dir}/embedded_tag_features.pth')
    features = torch.concatenate([embedded_img_features, embedded_tag_features], dim=0)
    features = features.to('cpu').detach().numpy().copy()

    # tSENで埋め込み
    tSNE_feature_filename = f'{SAVE_DIR}/tSNE_feature.npz'
    if os.path.exists(tSNE_feature_filename):
        tSNE_embedding = np.load(tSNE_feature_filename)['arr_0']
        print("Loaded existing t-SNE feature.")
    else:
        print(f"tSNE transform start ({DATASET})")
        tSNE_embedding = tSNE.transform(features)
        np.savez_compressed(tSNE_feature_filename, tSNE_embedding)
        print(f"tSNE transform end ({DATASET})")
        print("Calculated and saved new t-SNE.")
    X = tSNE_embedding[:, 0]
    Y = tSNE_embedding[:, 1]

    # モダリティ(画像/印象)で色分けしてプロット
    fig, ax = plt.subplots(figsize=(6.4*1.5, 4.8*1.5))
    modality = ['img', 'tag']
    patches = [mpatches.Patch(color=plt.cm.tab10(i), label=modality[i]) for i in range(2)]
    labels = [0]*len(embedded_img_features) + [1]*len(embedded_tag_features)
    sc = plt.scatter(X, Y, c=plt.cm.tab10(np.asarray(labels, dtype=np.int64)), alpha=0.8, edgecolors='w',
                    linewidths=0.1, s=5)
    plt.savefig(f"{SAVE_DIR}/tSNE.png", bbox_inches='tight', dpi=500)
    plt.legend()
    plt.close()


    # ラベル(クラスタID)の取得
    img_cluster_id = np.load(IMG_CLUSTER_PATH)["arr_0"].astype(np.int64)
    tag_cluster_id = np.load(TAG_CLUSTER_PATH)["arr_0"].astype(np.int64)

    # 各モダリティのクラスタ別に色分け (学習データのみ)
    if DATASET=='train':
        for ANNOTATE_WITH in ['img', 'tag']:
            # マウスオーバーで画像とクラス，ファイル名を表示
            fig, ax = plt.subplots(figsize=(6.4*1.5, 4.8*1.5))
            img_paths, tag_paths = utils.load_dataset_paths(DATASET)
            img = np.load(img_paths[0])["arr_0"][0]
            imagebox = OffsetImage(img, zoom=0.7, cmap='gray')
            imagebox.image.axes = ax

            if ANNOTATE_WITH=='img':
                labels = list(img_cluster_id*2)+list(img_cluster_id*2+1)
            elif ANNOTATE_WITH=='tag':
                labels = list(tag_cluster_id*2)+list(tag_cluster_id*2+1)
            modality = ['img', 'tag']
            patches = [mpatches.Patch(color=plt.cm.tab20(i), label=f"cluster{i//2}_{modality[i%2]}") for i in range(20)]
            sc = plt.scatter(X, Y, c=plt.cm.tab20(np.asarray(labels, dtype=np.int64)), 
                             alpha=0.8, edgecolors='w', linewidths=0.1, s=5)
            
            annot_img = AnnotationBbox(imagebox, xy=(0,0), xycoords="data", boxcoords="offset points", pad=0,
                                    arrowprops=dict( arrowstyle="->", connectionstyle="arc3,rad=-0.3"))
            annot_img.set_visible(False)
            ax.add_artist(annot_img)

            annot_text = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points", bbox=dict(boxstyle="round", fc="w"))
            annot_text.set_visible(False)

            fig.canvas.mpl_connect("motion_notify_event", hover)
            # plt.legend(handles=patches)
            plt.savefig(f"{SAVE_DIR}/tSNE_{ANNOTATE_WITH}.png", bbox_inches='tight', dpi=500)
            # plt.show()
            plt.close()