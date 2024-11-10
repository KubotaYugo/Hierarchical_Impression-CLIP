import torch
import os
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from models import FontAutoencoder
from lib import utils
from lib import bisecting_kmeans


def plot_scores(score, filename):
    plt.figure(figsize=(8, 5))
    plt.plot([i+2 for i in range(len(score))], score)
    plt.xlabel('Number of clusters')
    plt.ylabel(filename)
    plt.grid(True)
    plt.savefig(f'{SAVE_DIR}/{filename}.png', bbox_inches='tight', dpi=300)
    plt.close()


# ハイパラ
EXP = utils.EXP
DATASET = "train"
AUTOENCODER_PATH = "FontAutoencoder/model/best.pt"
BATCH_SIZE = 256

NUMBER_OF_CLUSTERS_MAX = 150

SAVE_DIR = f'{EXP}/quantitative_evaluation_standardization_img/{DATASET}'
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


# 評価指標の計算
SSE_list = []
silhouette_lsit = []
davies_bouldin_list = []
calinski_harabasz_list = []
for NUMBER_OF_CLUSTERS in range(2, NUMBER_OF_CLUSTERS_MAX+1): 
    print(NUMBER_OF_CLUSTERS)
    # get labels
    labels = np.load(f'{EXP}/clustering_standardized_img/{DATASET}/{NUMBER_OF_CLUSTERS}.npz')["arr_0"].astype(np.int16)
    # calculate scores
    SSE = bisecting_kmeans.calculate_inertia(stacked_img_features, labels)
    silhouette = silhouette_score(stacked_img_features, labels)
    davies_bouldin = davies_bouldin_score(stacked_img_features, labels)
    calinski_harabasz = calinski_harabasz_score(stacked_img_features, labels)
    # append scores to lists
    SSE_list.append(SSE)
    silhouette_lsit.append(silhouette)
    davies_bouldin_list.append(davies_bouldin)
    calinski_harabasz_list.append(calinski_harabasz)

plot_scores(SSE_list, 'SSE')
plot_scores(silhouette_lsit, 'silhouette')
plot_scores(davies_bouldin_list, 'davies_bouldin')
plot_scores(calinski_harabasz_list, 'calinski_harabasz')