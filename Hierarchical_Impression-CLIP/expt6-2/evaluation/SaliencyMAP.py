'''
画像に対してGradCAMをする
1. テキストから画像を検索
2. そのテキストに対する画像の勾配を得る
3. 勾配を可視化
'''

# チャンネル毎の平均を見れば，どの文字をヒントに見ているかがわかる
# セリフという印象に対する'O'の勾配は小さい，ということがあるかも

import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPModel
import matplotlib.pyplot as plt

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils
from models import MLP
from models import HierarchicalDataset
from models import FontAutoencoder


# define constant
params = utils.get_parameters()
DATASET = params.dataset
MODEL_PATH = params.model_path
FONTAUTOENCODER_PATH = params.fontautoencoder_path
EMBEDDED_SINGLE_TAG_FEATURE_PATH = params.embedded_single_tag_feature_path


# データの準備
# img_paths, _ = utils.load_dataset_paths(DATASET)
fontname = 'larque'
img_path = f"dataset/MyFonts_preprocessed/font_numpy_Impression-CLIP/{DATASET}/{fontname}.npz"
# 画像
tag = 'formal'
tag_list = utils.get_tag_list()
tag_num = np.where(tag_list==tag)[0][0]
input_img = np.load(img_path)['arr_0'].astype(np.float32)
input_img = torch.from_numpy(input_img/255).to('cuda')
input_img = input_img.unsqueeze(0)
input_img.requires_grad_()
# 印象
embedded_single_tag_feature = torch.load(EMBEDDED_SINGLE_TAG_FEATURE_PATH)

# モデルの準備
device = torch.device('cuda:0')
font_autoencoder = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device)
font_autoencoder.load_state_dict(torch.load(FONTAUTOENCODER_PATH))
font_encoder = font_autoencoder.encoder
emb_img = MLP.ReLU().to(device)
emb_img.load_state_dict(torch.load(MODEL_PATH)['emb_img'])

# cos類似度を計算
img_feature = font_encoder(input_img)
embedded_img_feature = emb_img(img_feature)
similarlity = torch.matmul(embedded_img_feature, embedded_single_tag_feature[tag_num].T)
print(tag_list[tag_num], similarlity.item())

# score, class_idx = torch.max(similarlity, 1)
score = similarlity
font_encoder.zero_grad()
emb_img.zero_grad()
score.backward()

saliency = input_img.grad.data.abs().to('cpu').detach().numpy()

input_img = np.load(img_path)['arr_0'].astype(np.float32)
heatmap_list = []
img_list = []
for i in range(26):
    heatmap = np.maximum(saliency[0][i], 0)
    # heatmap = saliency[0][i]
    heatmap_list.append(heatmap)
    img = input_img[i]
    img_list.append(img)

concatenated_heatmap = np.hstack(heatmap_list)
concatenated_img = np.hstack(img_list)
concatenated_heatmap = concatenated_heatmap / concatenated_heatmap.max()
concatenated_heatmap = np.uint8(255 * concatenated_heatmap)  # 0-255の範囲に変換
concatenated_heatmap = plt.get_cmap('jet')(concatenated_heatmap)
gray_image_rgb = np.repeat(concatenated_img[:, :, np.newaxis], 3, axis=2)
overlay = np.uint8(0.5 * concatenated_heatmap[:, :, :3]*255 + 0.5 * gray_image_rgb)
plt.imshow(overlay)
plt.axis('off')
plt.show()


# saliency, _ = torch.max(input_img.grad.data.abs(), dim=1)