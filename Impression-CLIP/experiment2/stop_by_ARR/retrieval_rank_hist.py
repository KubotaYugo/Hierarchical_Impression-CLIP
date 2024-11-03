import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt

import os
import utils
import eval_utils


# define constant
EXP = utils.EXP
BATCH_SIZE = utils.BATCH_SIZE    # フォントの数 (画像と印象を合わせた数はこれの2倍)
DATASET = 'train'
MODEL_PATH = f'{EXP}/results/model/best.pth.tar'

SAVE_DIR = f'{EXP}/retrieval_rank_hist/{DATASET}'
os.makedirs(SAVE_DIR, exist_ok=True)

# 特徴量の読み込み
load_dir = f'{EXP}/features/{DATASET}'
embedded_img_features = torch.load(f'{load_dir}/embedded_img_features.pth')
embedded_tag_features = torch.load(f'{load_dir}/embedded_tag_features.pth')

# Retrieval Rankの計算
similarity_matrix = torch.matmul(embedded_img_features, embedded_tag_features.T)
RR_tag2img = eval_utils.retrieval_rank(similarity_matrix, "tag2img")
RR_img2tag = eval_utils.retrieval_rank(similarity_matrix, "img2tag")

# タグの個数を取得
_, tag_paths = utils.LoadDatasetPaths(DATASET)
number_of_tags = [len(utils.get_font_tags(tag_path)) for tag_path in tag_paths]
number_of_tags = np.asarray(number_of_tags)


def plot_retrival_rank(number_of_tags, RR, mode, SAVE_DIR):
    # ヒストグラムの階級の幅を設定
    number_of_bins = 30
    bin_width = np.ceil(max(RR)/number_of_bins)
    bins = np.arange(1, max(RR)+bin_width, bin_width)
    RR = np.asarray(RR)

    # RRのヒストグラム
    fig, ax = plt.subplots(figsize=(7, 4.8))
    plt.hist(RR, bins=bins, edgecolor='black')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f"{SAVE_DIR}/{mode}.png", dpi=500, bbox_inches='tight')
    plt.close()

    # RRのヒストグラム(タグの個数別に色分け)
    data = [RR[number_of_tags==i] for i in range(1, 10+1)]
    labels = [f'{i}' for i in range(1, 10+1)]
    plt.hist(data, bins=bins, stacked=True, edgecolor='black', label=labels)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.savefig(f"{SAVE_DIR}/{mode}_with_number_of_tags.png", dpi=500, bbox_inches='tight')
    plt.close()

    # タグの個数毎に別々にヒストグラムを折れ線でプロット
    for i in range(1, 10+1):
        fig, ax = plt.subplots(figsize=(7, 4.8))
        plt.hist(RR[number_of_tags==i], bins=bins, edgecolor='black')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(f"{SAVE_DIR}/{mode}_with_number_of_tags_{i}.png", dpi=500, bbox_inches='tight')
        plt.close()

    # タグの個数別に，(階級の度数)/(全体の数)で折れ線グラフ
    fig, ax = plt.subplots(figsize=(7, 4.8)) 
    for i in range(1, 10+1):
        counts, bin_edges = np.histogram(RR[number_of_tags==i], bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(bin_centers, counts/len(RR[number_of_tags==i]), '-o', label=f'{i}', markersize=2)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.savefig(f"{SAVE_DIR}/{mode}_with_number_of_tags_rate.png", dpi=500, bbox_inches='tight')
    # plt.show()
    plt.close()

# Retrieval Rankの頻度をプロット
plot_retrival_rank(number_of_tags, RR_tag2img, 'tag2img', SAVE_DIR)
plot_retrival_rank(number_of_tags, RR_img2tag, 'img2tag', SAVE_DIR)