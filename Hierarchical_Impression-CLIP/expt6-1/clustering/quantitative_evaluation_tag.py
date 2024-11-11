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
    plt.ylabel(filename)
    plt.grid(True)
    plt.savefig(f'{SAVE_DIR}/{filename}.png', bbox_inches='tight', dpi=300)
    plt.close()


# define constant
params = utils.get_parameters()
EXPT = params['expt']
DATASET = params['dataset']
TAG_FEATURE_PATH = params['tag_feature_path']
TAG_CLUSTER_PATH = params['tag_cluster_path']
NUM_TAG_CLUSTERS = params['num_tag_clusters']
TAG_PREPROCESS = params['tag_preprocess']

NUMBER_OF_CLUSTERS_MAX = 10

SAVE_DIR = f'{EXPT}/clustering/quantitative_evaluation_tag/{DATASET}/{TAG_PREPROCESS}/{NUM_TAG_CLUSTERS}'
os.makedirs(SAVE_DIR, exist_ok=True)

# 印象特徴の取得&標準化(train基準)
tag_feature_path_train = f'{EXPT}/tag_features/train.pth'
tag_features_train = torch.load(tag_feature_path_train).to("cpu").detach().numpy()
scaler = StandardScaler()
standardizer = scaler.fit(tag_features_train)
tag_features = torch.load(TAG_FEATURE_PATH).to("cpu").detach().numpy()
standardized_tag_features = standardizer.transform(tag_features)


# 評価指標の計算
SSE_list = []
silhouette_lsit = []
davies_bouldin_list = []
calinski_harabasz_list = []
for NUMBER_OF_CLUSTERS in range(2, NUMBER_OF_CLUSTERS_MAX+1): 
    print(NUMBER_OF_CLUSTERS)
    # get labels
    label_path = f'{EXPT}/clustering/cluster_tag/{DATASET}/{TAG_PREPROCESS}/{NUM_TAG_CLUSTERS}.npz'
    labels = np.load(label_path)["arr_0"].astype(np.int16)
    # calculate scores
    SSE = bisecting_kmeans.calculate_inertia(standardized_tag_features, labels)
    silhouette = silhouette_score(standardized_tag_features, labels)
    davies_bouldin = davies_bouldin_score(standardized_tag_features, labels)
    calinski_harabasz = calinski_harabasz_score(standardized_tag_features, labels)
    # append scores to lists
    SSE_list.append(SSE)
    silhouette_lsit.append(silhouette)
    davies_bouldin_list.append(davies_bouldin)
    calinski_harabasz_list.append(calinski_harabasz)

plot_scores(SSE_list, 'SSE')
plot_scores(silhouette_lsit, 'silhouette')
plot_scores(davies_bouldin_list, 'davies_bouldin')
plot_scores(calinski_harabasz_list, 'calinski_harabasz')