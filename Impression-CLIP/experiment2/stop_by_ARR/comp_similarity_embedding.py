'''
画像/印象それぞれのモダリティ内で以下の2軸を持つ頻度のヒートマップを作成
横軸: 埋め込み前の画像(印象)特徴どうしの距離
縦軸: 埋め込み後の画像(印象)特徴どうしの類似度
'''
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import utils


def plot_heatmap(distance, embedded_similarity, filename):
    BIN_NUM = 50
    TICKS_NUM = 11

    # 2次元配列の非対角成分を1次元配列に
    mask = ~torch.eye(distance.size(0), dtype=bool)
    distance = distance[mask].to('cpu').detach().numpy().copy()
    embedded_similarity = embedded_similarity[mask].to('cpu').detach().numpy().copy()

    # ヒストグラムの範囲を値域に応じて設定
    x_range = [distance.min(), distance.max()]
    y_range = [embedded_similarity.min(), embedded_similarity.max()]
    heatmap, _, _ = np.histogram2d(distance, embedded_similarity, bins=BIN_NUM, range=[x_range, y_range])

    ax = sns.heatmap(heatmap, cmap='coolwarm')
    ax.set_xlabel('Distance before embedding')
    ax.set_ylabel('Similarity after embedding')

    # 軸のメモリの設定
    xtick_pos = np.linspace(0, heatmap.shape[1], TICKS_NUM)
    ytick_pos = np.linspace(0, heatmap.shape[0], TICKS_NUM)
    ax.set_xticks(xtick_pos)
    ax.set_yticks(ytick_pos)

    xticklabels = np.int64(np.linspace(distance.min(), distance.max(), TICKS_NUM).round(0))
    yticklabels = np.linspace(embedded_similarity.min(), embedded_similarity.max(), TICKS_NUM).round(3)
    ax.set_xticklabels(xticklabels, rotation=0)
    ax.set_yticklabels(yticklabels)
    
    plt.savefig(f'{SAVE_DIR}/{filename}.png', dpi=500, bbox_inches='tight')
    plt.close()


# define constant
EXP = utils.EXP

for DATASET in ['train', 'val', 'test']:
    SAVE_DIR = f'{EXP}/comp_similarity_embedding/{DATASET}'
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 特徴量の読み込み
    load_dir = f'{EXP}/features/{DATASET}'
    img_features = torch.load(f'{load_dir}/img_features.pth')
    tag_features = torch.load(f'{load_dir}/tag_features.pth')
    embedded_img_features = torch.load(f'{load_dir}/embedded_img_features.pth')
    embedded_tag_features = torch.load(f'{load_dir}/embedded_tag_features.pth')

    # 埋め込み前の画像/印象どうしの距離
    img_distance = torch.cdist(img_features, img_features, p=2)
    tag_distance = torch.cdist(tag_features, tag_features, p=2)

    # 埋め込み後の画像/印象どうしの類似度
    embedded_img_similarity = torch.matmul(embedded_img_features, embedded_img_features.T)
    embedded_tag_similarity = torch.matmul(embedded_tag_features, embedded_tag_features.T)

    plot_heatmap(img_distance, embedded_img_similarity, 'img')
    plot_heatmap(tag_distance, embedded_tag_similarity, 'tag')