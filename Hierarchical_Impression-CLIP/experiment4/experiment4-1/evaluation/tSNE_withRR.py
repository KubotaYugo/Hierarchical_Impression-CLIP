'''
Retrieval Rankの大小で色付けして画像特徴と印象特徴を別々にプロット
'''
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from transformers import CLIPTokenizer, CLIPModel
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from openTSNE import TSNE
from sklearn.decomposition import PCA
import pandas as pd

from models import FontAutoencoder
from models import MLP
from lib import utils
from lib import eval_utils



# define constant
EXP = utils.EXP
LR = utils.LR
BATCH_SIZE = utils.BATCH_SIZE
MODEL_PATH = f"{EXP}/LR={LR}, BS={BATCH_SIZE}/results/model/best.pth.tar"
DATASET = 'train'
SAVE_DIR = f"{EXP}/LR={LR}, BS={BATCH_SIZE}/tSNE_withRR/{DATASET}"

# 保存用フォルダの準備
os.makedirs(f"{SAVE_DIR}", exist_ok=True)

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

# 特徴量の取得
_, _, embedded_img_features, embedded_tag_features = eval_utils.extract_features(font_autoencoder, clip_model, emb_i, emb_t, dataloader)
# Retrieval Rankの計算
similarity_matrix = torch.matmul(embedded_img_features, embedded_tag_features.T)
RR_img2tag = eval_utils.retrieval_rank(similarity_matrix, "img2tag")
RR_tag2img = eval_utils.retrieval_rank(similarity_matrix, "tag2img")

# tSNE
tSNE_filename = f"{SAVE_DIR}/tSNE_embedding.npz"
if os.path.exists(tSNE_filename):
    embedding = np.load(tSNE_filename)['arr_0']
    print("Loaded existing t-SNE results.")
else:
    embedded_img_features = embedded_img_features.to('cpu').detach().numpy().copy()
    embedded_tag_features = embedded_tag_features.to('cpu').detach().numpy().copy()
    features = np.concatenate([embedded_img_features, embedded_tag_features], axis=0)
    PERPLEXITY = 30
    N_ITER = 300
    print("tSNE_start")
    embedding = TSNE(perplexity=PERPLEXITY, n_iter=N_ITER, initialization="pca", metric="euclidean", n_jobs=10, random_state=7).fit(features)
    np.savez_compressed(tSNE_filename, embedding)
    print("tSNE_end")
    print("Calculated and saved new t-SNE results.")


# RR_img2tag, RR_tag2imgで色付けした，画像特徴，印象特徴の分布をプロット
def plot(x, y, color, name):
    fig, ax = plt.subplots(figsize=(6.4*1.5, 4.8*1.5))
    sort_indices = np.argsort(np.abs(color-np.mean(color)))
    x = x[sort_indices]
    y = y[sort_indices]
    color = np.asarray(color)
    color = color[sort_indices]
    scatter = plt.scatter(x, y, c=color, alpha=0.8, edgecolors='w', linewidths=0.1, s=5, cmap='jet')
    plt.colorbar(scatter)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.savefig(f"{SAVE_DIR}/{name}.png", bbox_inches='tight', dpi=500)
    # plt.show()
    plt.close()

x = embedding[:, 0]
y = embedding[:, 1]
image_x = x[:len(img_paths)]
image_y = y[:len(img_paths)]
tag_x = x[len(img_paths):]
tag_y = y[len(img_paths):]
x_min = min(x)-5
x_max = max(x)+5
y_min = min(y)-5
y_max = max(y)+5
plot(image_x, image_y, RR_img2tag, 'img2tag_img')
plot(image_x, image_y, RR_tag2img, 'tag2img_img')
plot(tag_x, tag_y, RR_img2tag, 'img2tag_tag')
plot(tag_x, tag_y, RR_tag2img, 'tag2img_tag')


# RR_img2tag, RR_tag2imgの差分で色付けした，画像特徴，印象特徴の分布をプロット
RR_img2tag = np.asarray(RR_img2tag)
RR_tag2img = np.asarray(RR_tag2img)
plot(image_x, image_y, RR_img2tag-RR_tag2img, 'RR_img2tag-RR_tag2img_img')
plot(tag_x, tag_y, RR_tag2img-RR_img2tag, 'RR_tag2img-RR_img2tag_tag')

# インデックスで色付けした画像特徴，印象特徴の分布をプロット
img_distance_from_top_left = np.sqrt(pow(-80-image_x,2)+pow(80-image_y,2))
tag_distance_from_top_left = np.sqrt(pow(-80-tag_x,2)+pow(80-tag_y,2))

plot(image_x, image_y, image_x, 'index_img_x_img')
plot(tag_x, tag_y, image_x, 'index_img_x_tag')
plot(image_x, image_y, tag_x, 'index_tag_x_img')
plot(tag_x, tag_y, tag_x, 'index_tag_x_tag')

plot(image_x, image_y, image_y, 'index_img_y_img')
plot(tag_x, tag_y, image_y, 'index_img_y_tag')
plot(image_x, image_y, tag_y, 'index_tag_y_img')
plot(tag_x, tag_y, tag_y, 'index_tag_y_tag')

plot(image_x, image_y, img_distance_from_top_left, 'index_img_c_img')
plot(tag_x, tag_y, img_distance_from_top_left, 'index_img_c_tag')
plot(image_x, image_y, tag_distance_from_top_left, 'index_tag_c_img')
plot(tag_x, tag_y, tag_distance_from_top_left, 'index_tag_c_tag')