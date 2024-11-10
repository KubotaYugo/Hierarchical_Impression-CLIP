import torch
import os
from torch.autograd import Variable
import numpy as np
from transformers import CLIPTokenizer, CLIPModel
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
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
BATCH_SIZE = 256

NUMBER_OF_CLUSTERS_MAX = 100
N_ITER = 100  # bisecting kmeansを異なる初期値で繰り返す回数

SAVE_DIR = f'{EXP}/quantitative_evaluation_tag'
os.makedirs(SAVE_DIR, exist_ok=True)

# データの準備
_, tag_paths = utils.load_dataset_paths(DATASET)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
dataset = utils.ImpressionDataset(tag_paths, tokenizer)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = os.cpu_count(), pin_memory=True)

# モデルの準備
device = "cuda"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()

# 印象特徴の取得
with torch.no_grad():
    for i, data in enumerate(dataloader):
        tokenized_text = Variable(data).to(device)
        tag_feature =  model.get_text_features(tokenized_text)
        if i==0:
            stacked_tag_features = tag_feature
        else:
            stacked_tag_features = torch.concatenate((stacked_tag_features, tag_feature), dim=0)
stacked_tag_features = stacked_tag_features.to("cpu").detach().numpy()


# 評価指標の計算
SSE_list = []
silhouette_lsit = []
davies_bouldin_list = []
calinski_harabasz_list = []
for NUMBER_OF_CLUSTERS in range(2, NUMBER_OF_CLUSTERS_MAX+1): 
    print(NUMBER_OF_CLUSTERS)
    # get labels
    labels = np.load(f'{EXP}/clustering_img/{DATASET}/{NUMBER_OF_CLUSTERS}.npz')["arr_0"].astype(np.int16)
    # calculate scores
    SSE = bisecting_kmeans.calculate_inertia(stacked_tag_features, labels)
    silhouette = silhouette_score(stacked_tag_features, labels)
    davies_bouldin = davies_bouldin_score(stacked_tag_features, labels)
    calinski_harabasz = calinski_harabasz_score(stacked_tag_features, labels)
    # append scores to lists
    SSE_list.append(SSE)
    silhouette_lsit.append(silhouette)
    davies_bouldin_list.append(davies_bouldin)
    calinski_harabasz_list.append(calinski_harabasz)

plot_scores(SSE_list, 'SSE')
plot_scores(silhouette_lsit, 'silhouette')
plot_scores(davies_bouldin_list, 'davies_bouldin')
plot_scores(calinski_harabasz_list, 'calinski_harabasz')