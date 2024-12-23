'''
印象特徴間の類似度の分布を見る
'''
import torch
import matplotlib.pyplot as plt

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
TAG_PREPROCESS = params.tag_preprocess
NUMBER_OF_BINS = 50

# ディレクトリの作成
SAVE_DIR = f'{EXPT}/clustering/tag_feature_similarity/{TAG_PREPROCESS}'
os.makedirs(SAVE_DIR, exist_ok=True)

# 印象特徴の取得 & 類似度の計算
tag_features = torch.load(TAG_FEATURE_PATH)
tag_features_normalized = torch.nn.functional.normalize(tag_features, dim=1)
similarity_matrix = torch.matmul(tag_features_normalized, tag_features_normalized.T)
# 同じ印象特徴どうしの類似度は除く
similarity_matrix_non_diag_elements = similarity_matrix[~torch.eye(similarity_matrix.size(0), dtype=torch.bool)].view(-1)
similarity_matrix_non_diag_elements = similarity_matrix_non_diag_elements.to('cpu').detach().numpy()

# ヒストグラムの作成 
fig, ax = plt.subplots(figsize=(6.4*1.5, 4.8*1.5))
plt.hist(similarity_matrix_non_diag_elements, bins=NUMBER_OF_BINS, stacked=True)
plt.xlim(similarity_matrix_non_diag_elements.min(), similarity_matrix_non_diag_elements.max())
# plt.show()
plt.savefig(f'{SAVE_DIR}/{DATASET}.png', bbox_inches='tight', dpi=300)
plt.close()