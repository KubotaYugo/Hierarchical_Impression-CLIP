"""
AMT-setを使って, 人の目で見て最もその印象語らしいものを正しく上位にピックアップできるか
"""
import torch
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

import clip
import utils
import eval_utils

from pathlib import Path
DIR_PATH = Path(__file__).resolve().parent.parent.parent # /ICDAR_Kubotaまでのパスを取得

import sys
sys.path.append(f"{Path(__file__).resolve().parent.parent}")
import FontAutoencoder.FontAutoencoder as FontAutoencoder





def Plot(font_rank_list, text_rank_list, MODE):
    V_MIN = 0
    V_MAX = 5000
    V_CENTER = 2500
    FONT_SIZE = 16
    LABELS_FONT_SIZE = 11
    font_rank_list[font_rank_list==0] = 1
    text_rank_list[text_rank_list==0] = 1
    heatmap, xedges, yedges = np.histogram2d(font_rank_list, text_rank_list, bins=34, range=[[1,1709],[1,1709]])
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
    plt.savefig(f"{DIR_PATH}/Impression-CLIP/evaluate/comp_rank_matrix_{MODE}.png", dpi=600, bbox_inches="tight")
    plt.close()



#---------ハイパラ----------
K = 10
EPOCH = 10000
BATCH_SIZE = 8192
LEARNING_RATE = 1e-4
DATASET = "test"


#----------モデルの準備----------
device = torch.device('cuda:0')
font_autoencoder = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device)    
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
emb_f = utils.MLP().to(device)
emb_t = utils.MLP().to(device)
#-----パラメータのロード-----
font_autoencoder_state, clip_model_state, emb_f_state, emb_t_state, temp_state = list(torch.load(f"{DIR_PATH}/Impression-CLIP/model/best.pt").values())
font_autoencoder.load_state_dict(font_autoencoder_state)
clip_model.load_state_dict(clip_model_state)
emb_f.load_state_dict(emb_f_state)
emb_t.load_state_dict(emb_t_state)
#-----評価モードに-----
font_autoencoder.eval()
clip_model.eval()
emb_f.eval()
emb_t.eval()


#----------データの準備----------
font_paths, tag_paths = utils.LoadDatasetPaths(DATASET)
dataset = utils.CustomDataset(font_paths, tag_paths, clip.tokenize)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers = os.cpu_count(), pin_memory=True)

#-----タグ単体のdataloderの準備----
tag_list = list(eval_utils.GetTagList().values())    #評価対象とするタグのリスト
tagset = eval_utils.DatasetForTag(tag_list, clip.tokenize)
tagloader = torch.utils.data.DataLoader(tagset, batch_size=256, shuffle=False, num_workers = os.cpu_count(), pin_memory=True)



#----------フォントと印象をembedding----------
font_features, text_features, font_embedded_features, text_embedded_features = eval_utils.EmbedFontText(dataloader, font_autoencoder, clip_model, emb_f, emb_t, device)

#----------フォントと印象の類似度を計算-----------
similarity_matrix = torch.matmul(font_embedded_features, text_embedded_features.T).to("cpu").detach().numpy()

#----------埋め込み前後の検索順位----------
font_features = font_features.to('cpu').detach().numpy().copy()
text_features = text_features.to('cpu').detach().numpy().copy()
font_embedded_features = font_embedded_features.to('cpu').detach().numpy().copy()
text_embedded_features = text_embedded_features.to('cpu').detach().numpy().copy()

font_before_rank = eval_utils.CulculateRank(font_features, font_features)
text_before_rank = eval_utils.CulculateRank(text_features, text_features)
font_after_rank = eval_utils.CulculateRank(font_embedded_features, font_embedded_features)
text_after_rank = eval_utils.CulculateRank(text_embedded_features, text_embedded_features)

font_before_rank_list = font_before_rank[np.eye(font_before_rank.shape[0])==0]
text_before_rank_list = text_before_rank[np.eye(text_before_rank.shape[0])==0]
font_after_rank_list = font_after_rank[np.eye(font_after_rank.shape[0])==0]
text_after_rank_list = text_after_rank[np.eye(text_after_rank.shape[0])==0]



Plot(font_before_rank_list, text_before_rank_list, "before")
Plot(font_after_rank_list, text_after_rank_list, "after")