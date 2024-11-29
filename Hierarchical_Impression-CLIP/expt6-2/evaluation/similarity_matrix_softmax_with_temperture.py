'''
simlarity_matrixに温度パラメータ&ソフトマックスをかけて保存
...するといい感じに可視化されると思ったけど，微妙だったのでボツ(なにか考え直してみると面白いかも)
'''
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from models import ExpMultiplier
from lib import utils
import seaborn as sns


def save_similarity_matrix(embedded_img_feature=None, embedded_tag_feature=None, temperature=None, filename=None, ticks=None):
    # 温度パラメータをかけてソフトマックス
    similarity_matrix = temperature(torch.matmul(embedded_img_feature, embedded_tag_feature.T))
    softmax_tensor_x = np.log1p(F.softmax(similarity_matrix, dim=1).to("cpu").detach().numpy())
    softmax_tensor_y = np.log1p(F.softmax(similarity_matrix, dim=0).to("cpu").detach().numpy())
    similarity_matrix = F.softmax(similarity_matrix.view(-1), dim=0).view(similarity_matrix.size()).to("cpu").detach().numpy()
    similarity_matrix = similarity_matrix

    sns.heatmap(softmax_tensor_x, cmap='viridis', square=True, 
                cbar=True, vmin=np.min(softmax_tensor_x), vmax=np.max(softmax_tensor_x))
    plt.xlabel('Impression feature')
    plt.ylabel('Image feature')
    if ticks!=None:
        plt.xticks(ticks=ticks)
        plt.yticks(ticks=ticks) 
    plt.savefig(f'{filename}_x.png', dpi=300, bbox_inches='tight')
    plt.close()

    sns.heatmap(softmax_tensor_y, cmap='viridis', square=True, 
                cbar=True, vmin=np.min(softmax_tensor_y), vmax=np.max(softmax_tensor_y))
    plt.xlabel('Impression feature')
    plt.ylabel('Image feature')
    if ticks!=None:
        plt.xticks(ticks=ticks)
        plt.yticks(ticks=ticks) 
    plt.savefig(f'{filename}_y.png', dpi=300, bbox_inches='tight')
    plt.close()

    sns.heatmap(similarity_matrix, cmap='viridis', square=True, 
                cbar=True, vmin=np.min(similarity_matrix), vmax=np.max(similarity_matrix))
    plt.xlabel('Impression feature')
    plt.ylabel('Image feature')
    if ticks!=None:
        plt.xticks(ticks=ticks)
        plt.yticks(ticks=ticks) 
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()



# define constant
params = utils.get_parameters()
DATASET = params.dataset
BASE_DIR = params.base_dir
IMG_CLUSTER_PATH = params.img_cluster_path
TAG_CLUSTER_PATH = params.tag_cluster_path
EMBEDDED_IMG_FEATURE_PATH = params.embedded_img_feature_path
EMBEDDED_TAG_FEATURE_PATH = params.embedded_tag_feature_path
MODEL_PATH = params.model_path
TEMPERATURE = params.temperature

# ディレクトリの作成
SAVE_DIR = f'{BASE_DIR}/similarity_matrix_with_temperture/{DATASET}'
os.makedirs(SAVE_DIR, exist_ok=True)

# 温度パラメータ
params = torch.load(MODEL_PATH)
device = torch.device('cuda:0')
if TEMPERATURE=='ExpMultiplier':
    temperature = ExpMultiplier.ExpMultiplier().to(device)
elif TEMPERATURE=='ExpMultiplierLogit':
    temperature = ExpMultiplier.ExpMultiplierLogit().to(device)
temperature.load_state_dict(params['temperature'])

# 類似度行列の作成
embedded_img_feature = torch.load(EMBEDDED_IMG_FEATURE_PATH)
embedded_tag_feature = torch.load(EMBEDDED_TAG_FEATURE_PATH)
save_similarity_matrix(embedded_img_feature, embedded_tag_feature, temperature, f'{SAVE_DIR}/index')

# 画像のクラスタでソート
img_cluster = np.load(IMG_CLUSTER_PATH)["arr_0"].astype(np.int16())
number_of_instance = np.asarray([np.sum(img_cluster==i) for i in range(10)])
number_of_instance_cumulative = [np.sum(number_of_instance[:i+1]) for i in range(9)]
sorted_indices = np.argsort(img_cluster)
embedded_img_feature_sorted = embedded_img_feature[sorted_indices]
embedded_tag_feature_sorted = embedded_tag_feature[sorted_indices]
save_similarity_matrix(embedded_img_feature_sorted, embedded_tag_feature_sorted, temperature, 
                       f'{SAVE_DIR}/img_cluster', number_of_instance_cumulative)

# 印象のクラスタでソート
tag_cluster = np.load(TAG_CLUSTER_PATH)["arr_0"].astype(np.int16())
number_of_instance = np.asarray([np.sum(tag_cluster==i) for i in range(10)])
number_of_instance_cumulative = [np.sum(number_of_instance[:i+1]) for i in range(9)]
sorted_indices = np.argsort(tag_cluster)
embedded_img_feature_sorted = embedded_img_feature[sorted_indices]
embedded_tag_feature_sorted = embedded_tag_feature[sorted_indices]
save_similarity_matrix(embedded_img_feature_sorted, embedded_tag_feature_sorted, temperature, 
                       f'{SAVE_DIR}/tag_cluster', number_of_instance_cumulative)