import torch
import matplotlib.pyplot as plt
from openTSNE import TSNE

import pickle

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


# define constant
params = utils.get_parameters()
EXPT = params.expt
DATASET = params.dataset
TAG_FEATURE_PATH = params.tag_feature_path
TAG_CLUSTER_PATH = params.tag_cluster_path
NUM_TAG_CLUSTERS = params.num_tag_clusters

SAVE_DIR = f'{EXPT}/clustering/tSNE_single_tag'
os.makedirs(SAVE_DIR, exist_ok=True)

# タグ単体の印象特徴の読み込み
single_tag_feature_path = f'{EXPT}/feature/tag_feature/single_tag/train.pth'
single_tag_feature = torch.load(single_tag_feature_path).to('cpu').detach().numpy()

# tSNE
tSNE_filename = f'{SAVE_DIR}/tSNE_model.pkl'
if os.path.exists(tSNE_filename):
    with open(tSNE_filename, 'rb') as f:
        tSNE = pickle.load(f)
    print('Loaded existing tSNE model.')
else:
    print('tSNE start')
    tSNE = TSNE(initialization='pca', metric='euclidean', n_jobs=-1, random_state=7).fit(single_tag_feature)
    with open(tSNE_filename, 'wb') as f:
        pickle.dump(tSNE, f)
    print('tSNE end')
    print('Calculated and saved new tSNE.')

# 印象特徴の取得 & tSNE embedding
tSNE_feature_filename = f'{SAVE_DIR}/tSNE_feature.pkl'
if os.path.exists(tSNE_feature_filename):
    with open(tSNE_feature_filename, 'rb') as f:
        embedding = pickle.load(f)
    print('Loaded existing tSNE feature.')
else:
    print('tSNE embedding start')
    embedding = tSNE.transform(single_tag_feature)
    with open(tSNE_feature_filename, 'wb') as f:
        pickle.dump(embedding, f)
    print('tSNE embedding end')
    print('Calculated and saved new tSNE feature.')
X = embedding[:,0]
Y = embedding[:,1]

# プロット(マウスオーバーで画像と印象タグを表示)
tag_list = utils.get_tag_list()
fig, ax = plt.subplots()
fig.set_size_inches(6.4, 4.8)
plt.scatter(X, Y, s=0)
for x, y, tag in zip(X, Y, tag_list):
    plt.annotate(tag, xy=(x, y), fontsize=3)
# plt.savefig(f'{SAVE_DIR}/tSNE.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()