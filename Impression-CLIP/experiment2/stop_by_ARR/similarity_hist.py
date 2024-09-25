import os
from transformers import CLIPTokenizer, CLIPModel
import FontAutoencoder
import MLP
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import utils
import eval_utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_retrival_rank(RR, mode, SAVE_DIR):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    bin_width = 50
    bins = np.arange(1, max(RR)+bin_width, bin_width)
    plt.hist(RR, bins=bins, edgecolor='black')
    plt.xlim(0, 1750)
    plt.ylim(0, 700)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f"{SAVE_DIR}/retrieve_{mode}_rank_{DATASET}.png", dpi=300, bbox_inches='tight')


# define constant
EXP = utils.EXP
MODEL_PATH = f"{EXP}/model/best.pth.tar"
BATCH_SIZE = utils.BATCH_SIZE
DATASET = 'test'
SAVE_DIR = EXP

# モデルの準備
device = torch.device('cuda:0')
font_autoencoder = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device)
font_encoder = font_autoencoder.encoder    
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
emb_i = MLP.ReLU().to(device)
emb_t = MLP.ReLU().to(device)

# パラメータの読み込み
params = torch.load(MODEL_PATH)
font_encoder.load_state_dict(params['font_encoder'])
clip_model.load_state_dict(params['clip_model'])
emb_i.load_state_dict(params['emb_i'])
emb_t.load_state_dict(params['emb_t'])
font_autoencoder.eval()
clip_model.eval()
emb_i.eval()
emb_t.eval()

# dataloderの準備
img_paths, tag_paths = utils.LoadDatasetPaths(DATASET)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
dataset = utils.CustomDataset(img_paths, tag_paths, tokenizer)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)


# similarityの計算
_, _, embedded_img_features, embedded_tag_features = eval_utils.extract_features(font_autoencoder, clip_model, emb_i, emb_t, dataloader)
similarity_matrix = torch.matmul(embedded_img_features, embedded_tag_features.T).to("cpu").detach().numpy()

# 正例のsimilarityの頻度をプロット
fig, ax = plt.subplots(figsize=(7, 4.8))
bin_width = 0.1
bins = np.arange(-1, 1+bin_width, bin_width)
plt.hist(np.diag(similarity_matrix), bins=bins, edgecolor='black')
plt.savefig(f"{SAVE_DIR}/similarity_hist_positive_{DATASET}.png", bbox_inches='tight', dpi=300)

# 負例のsimilarityの頻度をプロット
fig, ax = plt.subplots(figsize=(7, 4.8))
bin_width = 0.1
bins = np.arange(-1, 1+bin_width, bin_width)
diag_mask = np.eye(similarity_matrix.shape[0], dtype=bool)
non_diag_mask = ~diag_mask
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.hist(similarity_matrix[non_diag_mask], bins=bins, edgecolor='black')
plt.savefig(f"{SAVE_DIR}/similarity_hist_negative_{DATASET}.png", bbox_inches='tight', dpi=300)