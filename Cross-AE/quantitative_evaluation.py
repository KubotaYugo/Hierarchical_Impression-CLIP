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
import utils


# ハイパラ
BATCH_SIZE = 128
DATASET = "test"
FONT_AUTOENCODER_STATE_PATH = "Cross-AE/co-training/img-AE/best(epoch=275).pt" 
# FONT_AUTOENCODER_STATE_PATH = "Cross-AE/co-training/img-AE/best(epoch=275).pt"  
IMP_AUTOENCODER_STATE_PATH = "Cross-AE/co-training/imp-AE/best(epoch=275).pt" 

# dataloder
img_dir = f"dataset/MyFonts_CompWithCrossAE/font_numpy/{DATASET}"
tag_dir = f"dataset/MyFonts_CompWithCrossAE/tag_txt/{DATASET}"
label_dir = "Cross-AE/impression-vector"
dataset = tools.FontDataloader2(img_dir, tag_dir, label_dir)
NUM_WORKERS = 0     #ここを0にしないと動かないかも
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# モデルの準備
device = 'cuda'
# img-AE
model_img = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device) 
model_img.load_state_dict(torch.load(FONT_AUTOENCODER_STATE_PATH, map_location='cuda:0'))
# imp-AE
model_imp = models.IntegratedModel(models.DeepSets(), models.imp_Encoder(), models.imp_Decoder()).to(device)
state_dict = torch.load(IMP_AUTOENCODER_STATE_PATH, map_location='cuda:0')
model_imp.load_state_dict(state_dict)

# Average Retrieval Rankの計算
img_features, tag_features = utils.extract_features(dataloader, model_img, model_imp, device)
rank_matrix, _ = utils.rank_matrix(img_features, tag_features)
sorted_index = np.argsort(rank_matrix, axis=1)
rank = [np.where(sorted_index[i]==i)[0].item()+1 for i in range(sorted_index.shape[0])]