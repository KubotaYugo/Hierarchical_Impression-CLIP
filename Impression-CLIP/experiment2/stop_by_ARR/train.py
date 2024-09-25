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
import MLP

# ハイパラ
MAX_EPOCH = 10000
BATCH_SIZE = 8192
LR = 1e-4
WARMUP_LR = 1e-5
EARLY_STOPPING_PATIENCE = 30

torch.cuda.empty_cache()

# 乱数の固定
utils.fix_seed(7)

# 保存用フォルダの準備 
SAVE_FOLDER = "Impression-CLIP/experiment2/stop_by_ARR"
os.makedirs(f"{SAVE_FOLDER}/model", exist_ok=True)
os.makedirs(f"{SAVE_FOLDER}/loss", exist_ok=True)

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
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
emb_i = MLP.ReLU().to(device)
emb_t = MLP.ReLU().to(device)
temp = utils.ExpMultiplier(initial_value=0.07).to(device)

# Autoencoderのパラメータを読み込む
FONT_AUTOENCODER_STATE_PATH = f"FontAutoencoder/model/best.pt" 
font_autoencoder.load_state_dict(torch.load(FONT_AUTOENCODER_STATE_PATH))
font_encoder = font_autoencoder.encoder

# オプティマイザ，loss, earlystopping, loggerの準備
models = [font_encoder, clip_model, emb_i, emb_t, temp]
parameters = list(emb_i.parameters())+list(emb_t.parameters())+list(temp.parameters())
optimizer = torch.optim.Adam(parameters, lr=LR)
criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
earlystopping = utils.EarlyStopping(patience=EARLY_STOPPING_PATIENCE, delta=0)
logger = SummaryWriter(f'{SAVE_FOLDER}/tensorboard')


train_losses = []
val_losses = []
val_losses_without_temp = []
temp_t_list = []
ARR_tag2img_list = []
ARR_img2tag_list = []
meanARR_list = []
meanARR_best = np.Inf
for epoch in tqdm(range(1, MAX_EPOCH+1)):
    # 学習&検証
    train_loss = utils.train(epoch, models, optimizer, criterion, trainloader, logger, device)
    val_loss, val_loss_without_temp, temp_t, ARR_tag2img, ARR_img2tag, meanARR = utils.val(epoch, models, criterion, valloader, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_losses_without_temp.append(val_loss_without_temp)
    temp_t_list.append(temp_t)
    ARR_tag2img_list.append(ARR_tag2img)
    ARR_img2tag_list.append(ARR_img2tag)
    meanARR_list.append(meanARR)

    logger.add_scalar('train_loss', train_loss, epoch)
    logger.add_scalar('val_loss', val_loss, epoch)
    logger.add_scalar('ARR_tag2img', ARR_tag2img, epoch)
    logger.add_scalar('ARR_img2tag', ARR_img2tag, epoch)
    logger.add_scalar('meanARR', meanARR, epoch)

    # 100エポック毎に保存
    if epoch%100==0:
        state = {"font_encoder":font_encoder.state_dict(), "clip_model":clip_model.state_dict(), "emb_i":emb_i.state_dict(), "emb_t":emb_t.state_dict(), "temp":temp.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, f"{SAVE_FOLDER}/model/epoch_{epoch:05d}.pth.tar")

    # loss_bestが更新されたら，そのときの状態を記録しておく
    if meanARR < meanARR_best:
        # モデルの保存
        best_font_encoder = copy.deepcopy(font_encoder.state_dict())
        best_clip_model = copy.deepcopy(clip_model.state_dict())
        best_temp = copy.deepcopy(temp.state_dict())
        best_optimizer = copy.deepcopy(optimizer.state_dict())
        # best_lossの更新とbest_epochの保存
        meanARR_best = meanARR
        best_epoch = epoch

    # early stoppingの判定
    if EARLY_STOPPING_PATIENCE != False:
        earlystopping(meanARR)
        if earlystopping.early_stop:
            print("Early stopping")
            break

# モデルと結果の保存
state = {"font_encoder":best_font_encoder, "clip_model":best_clip_model, "emb_i":emb_i.state_dict(), "emb_t":emb_t.state_dict(), "temp":best_temp, 'optimizer': best_optimizer}
torch.save(state, f"{SAVE_FOLDER}/model/best.pth.tar")


# csvへの保存
f = open(f"{SAVE_FOLDER}/loss/result.csv", 'w')
writer = csv.writer(f)
e_max = min(epoch, MAX_EPOCH)
for e in range(1-1, e_max+1-1):
    writer.writerow([e, train_losses[e], val_losses[e], val_losses_without_temp[e], temp_t_list[e], ARR_tag2img_list[e], ARR_img2tag_list[e], meanARR_list[e]])
f.close()