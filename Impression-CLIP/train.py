import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from transformers import CLIPTokenizer, CLIPModel
import utils

import FontAutoencoder
from warmup_cos_scheduler import CosineAnnealingLR as CosineAnnealingLR


# ハイパラ
MAX_EPOCH = 1000
BATCH_SIZE = 1024
LR = 1e-4
WARMUP_LR = 1e-5
EARLY_STOPPING_PATIENCE = 30

# 乱数の固定
utils.fix_seed(7)

# 保存用フォルダの準備 
os.makedirs(f"Impression-CLIP/model", exist_ok=True)
os.makedirs(f"Impression-CLIP/loss", exist_ok=True)

# データの準備
train_font_paths, train_tag_paths = utils.LoadDatasetPaths("train")
val_font_paths, val_tag_paths = utils.LoadDatasetPaths("val")
# N = 50
# BATCH_SIZE = 50
# train_font_paths = train_font_paths[:N]
# train_tag_paths = train_tag_paths[:N]
# val_font_paths = val_font_paths[:N]
# val_tag_paths = val_tag_paths[:N]
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
train_set = utils.CustomDataset(train_font_paths, train_tag_paths, tokenizer)
val_set = utils.CustomDataset(val_font_paths, val_tag_paths, tokenizer)
# 端数のバッチは学習に使わない
trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers = os.cpu_count(), pin_memory=True, drop_last=True)
valloader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers = os.cpu_count(), pin_memory=True)

# モデルの準備
device = torch.device('cuda:0')
font_autoencoder = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device)    
text_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").text_model.to(device)
temp = utils.ExpMultiplier(initial_value=0.07).to(device)

# Autoencoderのパラメータを読み込む
FONT_AUTOENCODER_STATE_PATH = f"FontAutoencoder/model/best.pt" 
font_autoencoder.load_state_dict(torch.load(FONT_AUTOENCODER_STATE_PATH))
font_encoder = font_autoencoder.encoder

# オプティマイザ，loss, earlystopping, loggerの準備
models = [font_encoder, text_encoder, temp]
parameters = list(font_encoder.parameters())+list(text_encoder.parameters())+list(temp.parameters())
optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=0.02)
lr_scheduler = CosineAnnealingLR(optimizer, max_epochs=MAX_EPOCH, warmup_epochs=32, warmup_start_lr=WARMUP_LR, eta_min=0.00001)
criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
earlystopping = utils.EarlyStopping(patience=EARLY_STOPPING_PATIENCE, delta=0)
logger = SummaryWriter('Impression-CLIP/tensorboard')


train_losses = []
val_losses = []
val_losses_without_temp = []
temp_t_list = []
val_loss_without_temp_best = np.Inf
for epoch in tqdm(range(1, MAX_EPOCH+1)):
    # 学習&検証
    train_loss = utils.train(epoch, models, optimizer, lr_scheduler, criterion, trainloader, logger, device)
    val_loss, val_loss_without_temp, temp_t = utils.val(epoch, models, criterion, valloader, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_losses_without_temp.append(val_loss_without_temp)
    temp_t_list.append(temp_t)

    logger.add_scalar('train_loss', train_loss, epoch)
    logger.add_scalar('val_loss', val_loss, epoch)

    # 100エポック毎に保存
    if epoch%100==0:
        state = {"font_encoder":font_encoder.state_dict(), "text_encoder":text_encoder.state_dict(), "temp":temp.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, "Impression-CLIP/model/epoch_{:05d}.pth.tar".format(epoch))

    # loss_bestが更新されたら，そのときの状態を記録しておく
    if val_loss_without_temp < val_loss_without_temp_best:
        # モデルの保存
        best_font_encoder = copy.deepcopy(font_encoder.state_dict())
        best_text_encoder = copy.deepcopy(text_encoder.state_dict())
        best_temp = copy.deepcopy(temp.state_dict())
        best_optimizer = copy.deepcopy(optimizer.state_dict())
        # best_lossの更新とbest_epochの保存
        val_loss_without_temp_best = val_loss_without_temp
        best_epoch = epoch

    # early stoppingの判定
    if EARLY_STOPPING_PATIENCE != False:
        earlystopping(val_loss_without_temp)
        if earlystopping.early_stop:
            print("Early stopping")
            break

# モデルと結果の保存
state = {"font_encoder":best_font_encoder, "text_encoder":best_text_encoder, "temp":best_temp, 'optimizer': best_optimizer}
torch.save(state, f"Impression-CLIP/model/best.pth.tar")


# csvへの保存
f = open(f"Impression-CLIP/loss/result.csv", 'w')
writer = csv.writer(f)
e_max = min(epoch, MAX_EPOCH)
for e in range(1-1, e_max+1-1):
    writer.writerow([e, train_losses[e], val_losses[e], val_losses_without_temp[e], temp_t_list[e]])
f.close() 