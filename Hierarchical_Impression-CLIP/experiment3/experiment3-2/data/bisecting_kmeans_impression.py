'''
画像特徴をbisecting kmeansでクラスタリング
'''

import torch
import os
from torch.autograd import Variable
import numpy as np
from transformers import CLIPTokenizer, CLIPModel

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


MODEL_PATH = "FontAutoencoder/model/best.pt"
DATASET = "train"

# データの準備
font_paths, tag_paths = utils.LoadDatasetPaths(DATASET)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
dataset = utils.ImpressionDataset(tag_paths, tokenizer)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers = os.cpu_count(), pin_memory=True)

# モデルの準備
device = "cuda"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)


# 印象特徴の取得
with torch.no_grad():
    for i, data in enumerate(dataloader):
        tokenized_text = Variable(data).to(device)
        text_feature =  model.get_text_features(tokenized_text)
        if i==0:
            text_features = text_feature.to("cpu").detach().numpy()
        else:
            text_features = np.concatenate([text_features, text_feature.to("cpu").detach().numpy()])


# bisecting kmeans
NUM_CLUSTERS = 10
clusters, data_index = utils.bisecting_kmeans(text_features, NUM_CLUSTERS)

# 表現形式を変えて, data_indexを保存
data_index_trasform = np.zeros(len(text_features))
for i in range(len(data_index)):
    data_index_trasform[data_index[i]] = i
# np.savez_compressed('imperession_clusters.npz', data_index_trasform)
# ↑誤って上書きしないようにコメントアウト