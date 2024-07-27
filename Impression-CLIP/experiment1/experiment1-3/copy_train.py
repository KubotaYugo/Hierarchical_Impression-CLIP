import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import clip
import copy_utils as utils

from pathlib import Path
DIR_PATH = Path(__file__).resolve().parent.parent.parent # /ICDAR_Kubotaまでのパスを取得

import sys
sys.path.append(f"{Path(__file__).resolve().parent.parent}")
import FontAutoencoder.FontAutoencoder as FontAutoencoder





#----------ハイパラ----------
EPOCH = 10000
BATCH_SIZE = 3
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 30

#---------乱数の固定----------
utils.fix_seed(7)

#---------保存用フォルダの準備----------
os.makedirs(f"{DIR_PATH}/Impression-CLIP/model", exist_ok=True)
os.makedirs(f"{DIR_PATH}/Impression-CLIP/loss", exist_ok=True)

#----------データの準備----------
train_font_paths, train_tag_paths = utils.LoadDatasetPaths("train")
val_font_paths, val_tag_paths = utils.LoadDatasetPaths("val")
train_set = utils.CustomDataset(train_font_paths, train_tag_paths, clip.tokenize)
val_set = utils.CustomDataset(val_font_paths, val_tag_paths, clip.tokenize)
#----------端数のバッチは学習に使わない----------
trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers = os.cpu_count(), pin_memory=True, drop_last=True)
valloader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers = os.cpu_count(), pin_memory=True)

#----------モデルの準備----------
device = torch.device('cuda:0')
font_autoencoder = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device)    
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
emb_f = utils.MLP().to(device)
emb_t = utils.MLP().to(device)
temp = utils.ExpMultiplier(initial_value=0.07).to(device)
models = [font_autoencoder, clip_model, emb_f, emb_t, temp]

#-----Autoencoderのパラメータを読み込む----------
FONT_AUTOENCODER_STATE_PATH = f"{DIR_PATH}/FontAutoencoder/model/best.pt" 
font_autoencoder.load_state_dict(torch.load(FONT_AUTOENCODER_STATE_PATH))

#----------オプティマイザ，loss, earlystoppingの準備----------
font_optimizer = optim.Adam(emb_f.parameters(), lr=LEARNING_RATE)
tag_optimizer = optim.Adam(emb_t.parameters(), lr=LEARNING_RATE)
temp_optimizer = optim.Adam(temp.parameters(), lr=LEARNING_RATE)
optimizers = [font_optimizer, tag_optimizer, temp_optimizer]
criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
earlystopping = utils.EarlyStopping(patience=EARLY_STOPPING_PATIENCE, delta=0)


for epoch in tqdm(range(0, EPOCH+1)):
    train_loss = utils.train(epoch, models, optimizers, criterion, trainloader, device)