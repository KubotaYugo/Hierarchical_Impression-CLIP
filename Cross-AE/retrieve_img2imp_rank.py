"""
画像から印象を検索したときの対応するペアの順位をまとめたCSVを出力
画像から印象を検索したときの対応するペアの順位の分布をグラフ化
"""
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import os
import matplotlib.pyplot as plt
import random
import numpy as np
import copy
import csv
from dataloader import DataLoader
import models
import tools
import FontAutoencoder


# ハイパラ
LEARNING_RATE = 0.001
BATCH_SIZE = 128
patience = 100
EPOCH = 5000

# 保存用ディレクトリの作成
SAVE_PATH = "Cross-AE/co-training/evaluate/retrieve_"
os.makedirs(f"{SAVE_PATH}",exist_ok=True)
os.makedirs(f"{SAVE_PATH}/imp-AE",exist_ok=True)
os.makedirs(f"{SAVE_PATH}/img-AE",exist_ok=True)

# dataloder
img_dir1 = "dataset/MyFonts_CompWithCrossAE/font_numpy/train"
img_dir2 = "dataset/MyFonts_CompWithCrossAE/font_numpy/val"
tag_dir1 = "dataset/MyFonts_CompWithCrossAE/tag_txt/train"
tag_dir2 = "dataset/MyFonts_CompWithCrossAE/tag_txt/val"
label_dir = "Cross-AE/impression-vector"
dataset_train = tools.FontDataloader2(img_dir1, tag_dir1, label_dir)
dataset_valid = tools.FontDataloader2(img_dir2, tag_dir2, label_dir)
NUM_WORKERS = 0     #ここを0にしないと動かないかも
loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
loader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# モデル, オプティマイザ, ロスの準備
device = 'cuda'
# img-AE
model_img = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device) 
FONT_AUTOENCODER_STATE_PATH = f"FontAutoencoder/model/best.pt" 
model_img.load_state_dict(torch.load(FONT_AUTOENCODER_STATE_PATH))
# imp-AE
model_imp = models.IntegratedModel(models.DeepSets(), models.imp_Encoder(), models.imp_Decoder()).to(device)
model_path = 'Cross-AE/imp-AE/model/best(epoch=122).pt'
state_dict = torch.load(model_path, map_location='cuda:0')
model_imp.load_state_dict(state_dict)
# optimizer, loss
parameters = list(model_img.parameters())+list(model_imp.parameters())
optimizer = optim.Adam(parameters, lr=LEARNING_RATE)
criterion = nn.MSELoss()


img_features, imp_features = utils.EmbedImgImp(dataloder, model_img, model_imp, device)
rank_matrix, _ = utils.RankMatrix(img_features, imp_features)

sorted_index = np.argsort(rank_matrix, axis=1)
rank = [np.where(sorted_index[i]==i)[0].item()+1 for i in range(sorted_index.shape[0])]



#----------csvに保存----------
font_names = utils.GetFontnames(DATASET)
with open(f"{DIR_PATH}/evaluate/retrieve_img2imp_rank_{DATASET}.csv", 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for i in range(len(font_names)):
        writer.writerow([font_names[i], rank[i]])

#----------順位の分布をプロット----------
fig, ax = plt.subplots(figsize=(7, 4.8))
bin_width = 10
bins = np.arange(1, max(rank)+bin_width, bin_width)
plt.hist(rank, bins=bins, edgecolor='black')
plt.xlim(0, 1750)
plt.ylim(0, 200)
plt.savefig(f"{DIR_PATH}/evaluate/retrieve_img2imp_rank_{DATASET}.png", dpi=300, bbox_inches='tight')

print(f"Average retrieval rank(img2imp) = {np.mean(rank)}")