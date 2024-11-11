import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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
    plt.ylabel('Sum of Squared Errors of prediction')
    plt.grid(True)
    plt.savefig(f'{SAVE_DIR}/{filename}.png', bbox_inches='tight', dpi=300)
    plt.close()


params = utils.get_parameters()
EXPT = params['expt']
BATCH_SIZE = params['batch_size']
DATASET = params['dataset']
IMG_FEATURE_PATH = params['img_feature_path']
IMG_CLUSTER_PATH = params['img_cluster_path']
NUM_IMG_CLUSTERS = params['num_img_clusters']

NUMBER_OF_CLUSTERS_MAX = 10

SAVE_DIR = f'{EXPT}/clustering/quantitative_evaluation_img/{DATASET}/{NUM_IMG_CLUSTERS}'
os.makedirs(SAVE_DIR, exist_ok=True)

# 画像特徴の取得&標準化(train基準)
img_feature_path_train = f'{EXPT}/img_features/train.pth'
img_features_train = torch.load(img_feature_path_train).to("cpu").detach().numpy()
scaler = StandardScaler()
standardizer = scaler.fit(img_features_train)
img_features = torch.load(IMG_FEATURE_PATH).to("cpu").detach().numpy()
standardized_img_features = standardizer.transform(img_features)

# 評価指標の計算
SSE_list = []
silhouette_lsit = []
davies_bouldin_list = []
calinski_harabasz_list = []
for NUMBER_OF_CLUSTERS in range(2, NUMBER_OF_CLUSTERS_MAX+1): 
    print(NUMBER_OF_CLUSTERS)
    # get labels
    label_path = f'{EXPT}/clustering/cluster_img/{DATASET}/{NUMBER_OF_CLUSTERS}.npz'
    labels = np.load(label_path)["arr_0"].astype(np.int16)
    # calculate scores
    SSE = bisecting_kmeans.calculate_inertia(standardized_img_features, labels)
    silhouette = silhouette_score(standardized_img_features, labels)
    davies_bouldin = davies_bouldin_score(standardized_img_features, labels)
    calinski_harabasz = calinski_harabasz_score(standardized_img_features, labels)
    # append scores to lists
    SSE_list.append(SSE)
    silhouette_lsit.append(silhouette)
    davies_bouldin_list.append(davies_bouldin)
    calinski_harabasz_list.append(calinski_harabasz)

plot_scores(SSE_list, 'SSE')
plot_scores(silhouette_lsit, 'silhouette')
plot_scores(davies_bouldin_list, 'davies_bouldin')
plot_scores(calinski_harabasz_list, 'calinski_harabasz')