'''
印象タグの個数で色分けして印象特徴をプロット
'''
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from openTSNE import TSNE
import utils
import eval_utils



# define constant
EXP = utils.EXP
DATASET = 'train'
SAVE_DIR = f"{EXP}/tSNE_withRR/{DATASET}"

# 保存用フォルダの準備
os.makedirs(f"{SAVE_DIR}", exist_ok=True)
# 特徴量の読み込み
load_dir = f'{EXP}/features/{DATASET}'
embedded_img_features = torch.load(f'{load_dir}/embedded_img_features.pth')
embedded_tag_features = torch.load(f'{load_dir}/embedded_tag_features.pth')

# Retrieval Rankの計算
similarity_matrix = torch.matmul(embedded_img_features, embedded_tag_features.T)
RR_img2tag = eval_utils.retrieval_rank(similarity_matrix, "img2tag")
RR_tag2img = eval_utils.retrieval_rank(similarity_matrix, "tag2img")

# tSNE
tSNE_filename = f"{EXP}/tSNE_withRR/{DATASET}/tSNE_embedding.npz"
if os.path.exists(tSNE_filename):
    embedding = np.load(tSNE_filename)['arr_0']
    print("Loaded existing t-SNE results.")
else:
    embedded_img_features = embedded_img_features.to('cpu').detach().numpy().copy()
    embedded_tag_features = embedded_tag_features.to('cpu').detach().numpy().copy()
    features = np.concatenate([embedded_img_features, embedded_tag_features], axis=0)
    PERPLEXITY = 30
    N_ITER = 300
    print("tSNE_start")
    embedding = TSNE(perplexity=PERPLEXITY, n_iter=N_ITER, initialization="pca", metric="euclidean", n_jobs=10, random_state=7).fit(features)
    np.savez_compressed(tSNE_filename, embedding)
    print("tSNE_end")
    print("Calculated and saved new t-SNE results.")

# タグの個数を取得
_, tag_paths = utils.LoadDatasetPaths(DATASET)
number_of_tags = [len(utils.get_font_tags(tag_path)) for tag_path in tag_paths]
number_of_tags = np.asarray(number_of_tags)

# タグの個数で色分けして，印象特徴をプロット
x = embedding[:, 0]
y = embedding[:, 1]
tag_x = x[len(embedded_img_features):]
tag_y = y[len(embedded_img_features):]
x_min = min(x)-5
x_max = max(x)+5
y_min = min(y)-5
y_max = max(y)+5

fig, ax = plt.subplots(figsize=(6.4*1.5, 4.8*1.5))
for i in range(1, 10+1):
    plt.scatter(tag_x[number_of_tags==i], tag_y[number_of_tags==i], alpha=0.8, edgecolors='w', linewidths=0.1, s=5, label=f'{i}')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend()
plt.savefig(f"{SAVE_DIR}/tSNE_with_number_of_tags.png", bbox_inches='tight', dpi=500)
plt.close()