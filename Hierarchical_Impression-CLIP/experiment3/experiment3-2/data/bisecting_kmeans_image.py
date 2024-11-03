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

MODEL_PATH = "FontAutoencoder/model/best.pt"
DATASET = "train"

# データの準備
font_paths, tag_paths = utils.LoadDatasetPaths(DATASET)
dataset = utils.ImageDataset(font_paths)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers = os.cpu_count(), pin_memory=True)

# モデルの準備
device = "cuda"
model = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# 画像特徴の取得
with torch.no_grad():
    for i, data in enumerate(dataloader):
        input_font = Variable(data).to(device)
        font_feature = model.encoder(input_font)
        if i==0:
            font_features = font_feature.to("cpu").detach().numpy()
        else:
            font_features = np.concatenate([font_features, font_feature.to("cpu").detach().numpy()])


# bisecting kmeans
NUM_CLUSTERS = 10
clusters, data_index = utils.bisecting_kmeans(font_features, NUM_CLUSTERS)

# 表現形式を変えて, data_indexを保存
data_index_trasform = np.zeros(len(font_features))
for i in range(len(data_index)):
    data_index_trasform[data_index[i]] = i
# np.savez_compressed('image_clusters.npz', data_index_trasform)
# ↑誤って上書きしないようにコメントアウト