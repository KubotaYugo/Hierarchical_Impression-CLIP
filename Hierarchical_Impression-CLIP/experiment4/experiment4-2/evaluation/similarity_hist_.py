import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


# define constant
BASE_DIR = utils.BASE_DIR

for DATASET in ['train', 'val', 'test']:
    SAVE_DIR = f'{BASE_DIR}/similarity_hist'
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 特徴量の読み込み & similarityの計算
    load_dir = f'{BASE_DIR}/features/{DATASET}'
    embedded_img_features = torch.load(f'{load_dir}/embedded_img_features.pth')
    embedded_tag_features = torch.load(f'{load_dir}/embedded_tag_features.pth')
    similarity_matrix = torch.matmul(embedded_img_features, embedded_tag_features.T).to("cpu").detach().numpy()

    # FigureとAxesを生成
    fig, ax1 = plt.subplots(figsize=(7, 4.8))
    bin_width = 0.1
    bins = np.arange(-1, 1+bin_width, bin_width)

    # 1つ目のヒストグラム（左軸）
    color1 = 'blue'
    ax1.hist(np.diag(similarity_matrix), bins=bins, color=color1, alpha=0.6)
    ax1.set_xlabel('Cosine similarity')
    ax1.set_ylabel('Positive pair', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # 2つ目のヒストグラム（右軸）
    ax2 = ax1.twinx()  # 同じx軸を共有する2つ目のy軸を作成
    color2 = 'red'
    diag_mask = np.eye(similarity_matrix.shape[0], dtype=bool)
    non_diag_mask = ~diag_mask
    ax2.hist(similarity_matrix[non_diag_mask], bins=bins, color=color2, alpha=0.6)
    ax2.set_ylabel('Negative pair', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # グラフの表示
    plt.savefig(f"{SAVE_DIR}/{DATASET}.png", bbox_inches='tight', dpi=300)