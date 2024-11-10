'''
画像特徴をbisecting kmeansでクラスタリング
'''

import torch
import os
from torch.autograd import Variable
import numpy as np

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from models import FontAutoencoder
from lib import utils
from lib import bisecting_kmeans


# ハイパラ
EXP = utils.EXP
DATASET = "train"
IMG_CLUSTER_PATH = utils.IMG_CLUSTER_PATH
AUTOENCODER_PATH = "FontAutoencoder/model/best.pt"
BATCH_SIZE = 256

NUMBER_OF_CLUSTERS = 10
N_ITER = 100  # bisecting kmeansを異なる初期値で繰り返す回数

SAVE_DIR = f'{EXP}/clustering/{DATASET}'
os.makedirs(SAVE_DIR, exist_ok=True)


# データの準備
img_paths, _ = utils.load_dataset_paths(DATASET)
dataset = utils.ImageDataset(img_paths)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = os.cpu_count(), pin_memory=True)

# モデルの準備
device = "cuda"
model = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device)
model.load_state_dict(torch.load(AUTOENCODER_PATH))
model.eval()

# 画像特徴の取得
with torch.no_grad():
    for i, data in enumerate(dataloader):
        input_img = Variable(data).to(device)
        img_feature = model.encoder(input_img)
        if i==0:
            stacked_img_features = img_feature
        else:
            stacked_img_features = torch.concatenate((stacked_img_features, img_feature), dim=0)
stacked_img_features = stacked_img_features.to("cpu").detach().numpy()


# bisecting kmeans
utils.fix_seed(666)
best_inertia = np.inf
inertia_list = []
for i in range(N_ITER):
    _, label = bisecting_kmeans.bisecting_kmeans(stacked_img_features, NUMBER_OF_CLUSTERS)
    inertia = bisecting_kmeans.calculate_inertia(stacked_img_features, label)
    print(f'inertia of iteration {i}: {inertia}')
    # inertia最小のクラスタリング結果を使用
    if inertia < best_inertia:
        print('renewed best labels')
        best_label = label
        best_inertia = inertia

# [2, 2, 1, 4, 5, 3, 2, 5, 4] -> [0, 0, 1, 2, 3, 4, 0, 3, 2]のように，リスト先頭から出てくる順に番号を振り直す
unique_numbers = {}
replaced_label = []
current_number = 0
for num in best_label:
    if num not in unique_numbers:
        unique_numbers[num] = current_number
        current_number += 1
    replaced_label.append(unique_numbers[num])

np.savez_compressed(IMG_CLUSTER_PATH, replaced_label)