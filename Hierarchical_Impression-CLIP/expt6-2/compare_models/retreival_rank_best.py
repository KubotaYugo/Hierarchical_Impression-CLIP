'''
横軸: モデル1のretrieval rank (画像 or 印象, シード値を通して平均)
縦軸: モデル1のretrieval rank (画像 or 印象, シード値を通して平均)
として各フォントをプロット
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils
from lib import eval_utils


def get_RR(params, param_set):
    # パラメータ設定
    params.loss_type = param_set[0]
    params.weights = param_set[1]
    params.random_seed = param_set[2]

    params.base_dir = f'{params.expt}/co-embedding/C=[{params.num_img_clusters}, {params.num_tag_clusters}]_{params.tag_preprocess}_{params.temperature}_{params.learn_temperature}_{params.initial_temperature}_{params.loss_type}_{params.ce_bce}_W={params.weights}_seed={params.random_seed}'
    params.embedded_img_feature_path = f'{params.base_dir}/feature/embedded_img_feature/{params.dataset}.pth'
    params.embedded_tag_feature_path = f'{params.base_dir}/feature/embedded_tag_feature/{params.dataset}.pth'
    EMBEDDED_IMG_FEATURE_PATH = params.embedded_img_feature_path
    EMBEDDED_TAG_FEATURE_PATH = params.embedded_tag_feature_path

    # 特徴量の読み込み
    embedded_img_feature = torch.load(EMBEDDED_IMG_FEATURE_PATH)
    embedded_tag_feature = torch.load(EMBEDDED_TAG_FEATURE_PATH)

    # Retrieval Rankの計算
    similarity_matrix = torch.matmul(embedded_img_feature, embedded_tag_feature.T)
    RR_tag2img = eval_utils.retrieval_rank(similarity_matrix, 'tag2img')
    RR_img2tag = eval_utils.retrieval_rank(similarity_matrix, 'img2tag')

    return {'tag2img':RR_tag2img, 'img2tag':RR_img2tag}

def plot(x, y, filename):
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

    def update_annot(ind):
        i = ind['ind'][0]
        pos = sc.get_offsets()[i]
        index = i%len(img_paths)
        fontname = img_paths[index][len(f'dataset/MyFonts_preprocessed/font_numpy_Impression-CLIP/{DATASET}/'):-4]
        gap = len(img_paths)/20
        annot_img.xy = (pos[0]+gap, pos[1]+gap)
        annot_text.xy = (pos[0]+gap, pos[1]-gap)
        img = np.load(img_paths[index])['arr_0'][0]
        imagebox.set_data(img)
        tags = utils.get_font_tags(tag_paths[index])
        ids = f'{img_cluster_id[i]}-{tag_cluster_id[i]}'
        annot_text.set_text(f'{fontname} {ids} {tags}')

    # プロット & カラーバーの作成
    # fig, ax = plt.subplots(figsize=(6.4, 4.8))
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    color = np.asarray(x)-np.asarray(y)
    plt.plot([0, len(x)], [0, len(x)], color='gray', linewidth=0.5, zorder=1)
    sc = plt.scatter(x, y, alpha=0.7, s=1, zorder=2)
    # sc = plt.scatter(x, y, c=color, alpha=0.7, s=1, zorder=2)
    # plt.colorbar(sc)
    plt.axis('equal')

    # 画像とタグを表示するための処理
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
    
    # 表示 & 保存
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()


# define constant
params = utils.get_parameters()
EXPT = params.expt
DATASET = params.dataset
BASE_DIR = params.base_dir
IMG_CLUSTER_PATH = params.img_cluster_path
TAG_CLUSTER_PATH = params.tag_cluster_path

# ラベル(クラスタID)の取得
img_cluster_id = np.load(IMG_CLUSTER_PATH)['arr_0'].astype(np.int64)
tag_cluster_id = np.load(TAG_CLUSTER_PATH)['arr_0'].astype(np.int64)

# loss_type, weights
param_set = [['average',   [1.0, 0.0, 0.0,], 1],
             ['average',   [1.0, 1.0, 1.0,], 3],
             ['iterative', [1.0, 1.0, 1.0,], 5],
             ['label_and', [1.0, 1.0, 1.0,], 5]]
model_name = ['baseline', 'average', 'iterative', 'label_and']

# ディレクトリの作成
SAVE_DIR = f'{EXPT}/compare_models/retrieval_rank_best/{DATASET}'

# retrieval rankの計算
# RR_list_org[i]: param_set[i]のRR_tag2imgとRR_img2tag
RR_list_org = [get_RR(params, param_set_i) for param_set_i in param_set]

# プロット
for direction in ['tag2img', 'img2tag']:
    os.makedirs(f'{SAVE_DIR}/{direction}', exist_ok=True)
    for i in range(len(param_set)):
        for j in range(len(param_set)):
                x = RR_list_org[i][direction]
                y = RR_list_org[j][direction]
                filename = f'{SAVE_DIR}/{direction}/{model_name[i]}_{model_name[j]}.png'
                plot(x, y, filename)