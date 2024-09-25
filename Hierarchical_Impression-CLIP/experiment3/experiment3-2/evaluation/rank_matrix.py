from transformers import CLIPTokenizer, CLIPModel
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from models import FontAutoencoder
from models import MLP
from lib import utils
from lib import eval_utils



def culc_rank(query_feature, key_feature):
    similarity_matrix = torch.matmul(query_feature, key_feature.T).to("cpu").detach().numpy()
    distances = -similarity_matrix
    nearest_idx = np.argsort(distances, axis=1)
    rank = np.zeros_like(nearest_idx)
    rows, cols = np.indices(nearest_idx.shape)
    rank[rows, nearest_idx] = cols
    return rank

def Plot(img_rank_list, tag_rank_list, MODE):
    V_MIN = 0
    V_MAX = 5000
    V_CENTER = 2500
    FONT_SIZE = 16
    LABELS_FONT_SIZE = 11
    img_rank_list[img_rank_list==0] = 1
    tag_rank_list[tag_rank_list==0] = 1
    heatmap, xedges, yedges = np.histogram2d(img_rank_list, tag_rank_list, bins=34, range=[[1,1709],[1,1709]])
    ax = sns.heatmap(heatmap, cmap='coolwarm', annot=False, fmt=".2f", vmin=V_MIN, vmax=V_MAX, center=V_CENTER, square=True)
    ax.invert_yaxis()
    ticks = [x for x in range(0, len(xedges), 2)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([math.ceil(xedges[i]) for i in range(0, len(xedges), 2)], rotation=90, fontsize=LABELS_FONT_SIZE)
    ax.set_yticklabels([math.ceil(yedges[i]) for i in range(0, len(xedges), 2)], rotation=0, fontsize=LABELS_FONT_SIZE)
    ax.figure.axes[-1].tick_params(axis='x', pad=0)
    ax.figure.axes[-1].tick_params(axis='y', pad=0)
    ax.set_xlabel('Image-based rank', fontsize=FONT_SIZE)
    ax.set_ylabel('Impression-based rank', fontsize=FONT_SIZE)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 1000, 2000, 3000, 4000, 5000])
    cbar.set_ticklabels([' 0', ' 1000', ' 2000', ' 3000', ' 4000', ' ≥5000'])
    plt.savefig(f"{SAVE_DIR}/{MODE}_{DATASET}.png", dpi=600, bbox_inches="tight")
    plt.close()



# define constant
EXP = utils.EXP
IMG_HIERARCHY_PATH = utils.IMG_HIERARCHY_PATH
TAG_HIERARCHY_PATH = utils.TAG_HIERARCHY_PATH
LR = utils.LR
BATCH_SIZE = utils.BATCH_SIZE    # フォントの数 (画像と印象を合わせた数はこれの2倍)
MODEL_PATH = f'{EXP}/LR={LR}, BS={BATCH_SIZE}/results/model/best.pth.tar'
DATASET = 'test'

SAVE_DIR = f'{EXP}/LR={LR}, BS={BATCH_SIZE}/rank_matrix'
os.makedirs(SAVE_DIR, exist_ok=True)

# モデルの準備
device = torch.device('cuda:0')
font_autoencoder = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
emb_i = MLP.ReLU().to(device)
emb_t = MLP.ReLU().to(device)
font_autoencoder.eval()
clip_model.eval()
emb_i.eval()
emb_t.eval()


# パラメータの読み込み
params = torch.load(MODEL_PATH)
font_autoencoder.load_state_dict(params['font_autoencoder'])
clip_model.load_state_dict(params['clip_model'])
emb_i.load_state_dict(params['emb_i'])
emb_t.load_state_dict(params['emb_t'])

# dataloderの準備
img_paths, tag_paths = utils.LoadDatasetPaths(DATASET)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
dataset = eval_utils.DMH_D_Eval(img_paths, tag_paths, tokenizer)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)


# 特徴量の抽出
img_features, tag_features, embedded_img_features, embedded_tag_features = eval_utils.extract_features(font_autoencoder, clip_model, emb_i, emb_t, dataloader)

# 埋め込み前の検索順位
img_rank_before = culc_rank(img_features, img_features)
tag_rank_before = culc_rank(tag_features, tag_features)
img_rank_before_list = img_rank_before[np.eye(img_rank_before.shape[0])==0]
tag_rank_before_list = tag_rank_before[np.eye(tag_rank_before.shape[0])==0]
Plot(img_rank_before_list, tag_rank_before_list, "before")

# 埋め込み後の検索順位
img_rank_after = culc_rank(embedded_img_features, embedded_img_features)
tag_rank_after = culc_rank(embedded_tag_features, embedded_tag_features)
img_rank_after_list = img_rank_after[np.eye(img_rank_after.shape[0])==0]
tag_rank_after_list = tag_rank_after[np.eye(tag_rank_after.shape[0])==0]
Plot(img_rank_after_list, tag_rank_after_list, "after")