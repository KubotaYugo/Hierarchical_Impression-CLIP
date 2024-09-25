from transformers import CLIPTokenizer, CLIPModel
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from models import FontAutoencoder
from models import MLP
from lib import utils
from lib import eval_utils


# define constant
EXP = utils.EXP
LR = utils.LR
BATCH_SIZE = utils.BATCH_SIZE    # フォントの数 (画像と印象を合わせた数はこれの2倍)
MODEL_PATH = f'{EXP}/LR={LR}, BS={BATCH_SIZE}/results/model/best.pth.tar'
DATASET = 'test'

SAVE_DIR = f'{EXP}/LR={LR}, BS={BATCH_SIZE}'
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
# 単体タグのdataloder
tag_list = list(eval_utils.get_tag_list().values())
tagset = eval_utils.DMH_D_ForTag(tag_list, tokenizer)
tagloader = torch.utils.data.DataLoader(tagset, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)


# similarityの計算
_, _, embedded_img_features, embedded_tag_features = eval_utils.extract_features(font_autoencoder, clip_model, emb_i, emb_t, dataloader)
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
plt.savefig(f"{SAVE_DIR}/similarity_hist_{DATASET}.png", bbox_inches='tight', dpi=300)