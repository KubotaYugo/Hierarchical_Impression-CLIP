from transformers import CLIPTokenizer, CLIPModel
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from models import FontAutoencoder
from models import MLP
from lib import utils
from lib import eval_utils


def plot_retrival_rank(RR, mode, SAVE_DIR):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    bin_width = 50
    bins = np.arange(1, max(RR)+bin_width, bin_width)
    plt.hist(RR, bins=bins, edgecolor='black')
    plt.xlim(0, 1750)
    plt.ylim(0, 800)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f"{SAVE_DIR}/retrieve_{mode}_rank_{DATASET}.png", dpi=300, bbox_inches='tight')


# define constant
EXP = utils.EXP
LR = utils.LR
BATCH_SIZE = utils.BATCH_SIZE    # フォントの数 (画像と印象を合わせた数はこれの2倍)
MODEL_PATH = f'{EXP}/LR={LR}, BS={BATCH_SIZE}/results/model/best.pth.tar'
SAVE_DIR = f"{EXP}/LR={LR}, BS={BATCH_SIZE}/retrieve"
DATASET = 'test'

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
# 単体タグのdataloder
tag_list = list(eval_utils.get_tag_list().values())
tagset = eval_utils.DMH_D_ForTag(tag_list, tokenizer)
tagloader = torch.utils.data.DataLoader(tagset, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)


# Retrieval Rankの計算
_, _, embedded_img_features, embedded_tag_features = eval_utils.extract_features(font_autoencoder, clip_model, emb_i, emb_t, dataloader)
similarity_matrix = torch.matmul(embedded_img_features, embedded_tag_features.T)
RR_tag2img = eval_utils.retrieval_rank(similarity_matrix, "tag2img")
RR_img2tag = eval_utils.retrieval_rank(similarity_matrix, "img2tag")

# Retrieval Rankの頻度をプロット
plot_retrival_rank(RR_tag2img, 'tag2img', SAVE_DIR)
plot_retrival_rank(RR_img2tag, 'img2tag', SAVE_DIR)