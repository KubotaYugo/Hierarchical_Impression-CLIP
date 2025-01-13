'''
各サンプルの各階層におけるロスを見てみる
横軸: 階層，縦軸: ロス
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils
import models.Temperature as Temperature


def sup_con_loss(logits, label):
    # for numerical stability
    logits_max = logits.max(dim=1, keepdim=True)[0]
    logits = logits - logits_max.detach()
    # compute log_prob
    exp_logits = torch.exp(logits) 
    sum_exp_logits = exp_logits.sum(dim=1, keepdim=True)
    loss_pos_neg = logits - torch.log(sum_exp_logits)
    loss_pos = loss_pos_neg*label
    # 1つのインスタンスに対する正例について平均
    loss_instance = -1 * loss_pos.sum(dim=1) / label.sum(dim=1)
    del logits_max, logits, exp_logits, sum_exp_logits, loss_pos_neg, loss_pos, label
    return loss_instance

def hierarchical_sup_con_loss(logit, clusterID, loss_type):
    layer_loss = []
    for i in range(len(clusterID[0])):
        print(f'階層: {i}')
        # ラベルの前処理
        clusterID_prune = np.array([s[:i + 1] for s in clusterID])
        labels = (clusterID_prune[:, None] == clusterID_prune[None, :]).astype(np.uint8)
        labels = torch.from_numpy(labels).to('cuda')

        # ロスの計算
        if loss_type == 'SupCon':
            loss_temp = sup_con_loss(logit, labels)
        elif loss_type == 'BCE':
            labels = labels.to(torch.float64)
            loss_temp = F.binary_cross_entropy(torch.sigmoid(logit), labels, reduction='none')
            loss_temp = loss_temp.mean(dim=1)
        layer_loss.append(loss_temp.to('cpu').detach().numpy())
        del clusterID_prune, labels, loss_temp
    return layer_loss

def calc_layer_loss(EMBEDDED_IMG_FEATURE_PATH, EMBEDDED_TAG_FEATURE_PATH, MODEL_PATH, LOSS_TYPE):
    # 特徴量の取得
    embedded_img_feature = torch.load(EMBEDDED_IMG_FEATURE_PATH)
    embedded_tag_feature = torch.load(EMBEDDED_TAG_FEATURE_PATH)

    # 温度パラメータの読み込み
    model_params = torch.load(MODEL_PATH)
    temperature = Temperature.Temperature().to('cuda')
    temperature.load_state_dict(model_params['temperature'])

    # 類似度行列の計算
    similarity_matrix = torch.matmul(embedded_img_feature, embedded_tag_feature.T)
    similarity_matrix_with_temperature = temperature(similarity_matrix)
    logits_per_img = similarity_matrix_with_temperature
    logits_per_tag = similarity_matrix_with_temperature.T

    # ロスの計算
    with torch.no_grad():
        layer_loss_img2tag = hierarchical_sup_con_loss(logits_per_img, tag_cluster_id, LOSS_TYPE)
        layer_loss_tag2img = hierarchical_sup_con_loss(logits_per_tag, img_cluster_id, LOSS_TYPE)
    layer_loss_img2tag = np.asarray(layer_loss_img2tag)
    layer_loss_tag2img = np.asarray(layer_loss_tag2img)

    return layer_loss_img2tag, layer_loss_tag2img


# define constant
params = utils.get_parameters()
DATASET = params.dataset
BASE_DIR = params.base_dir
MODEL_PATH = params.model_path
IMG_CLUSTER_PATH = params.img_cluster_path
TAG_CLUSTER_PATH = params.tag_cluster_path
EMBEDDED_IMG_FEATURE_PATH = params.embedded_img_feature_path
EMBEDDED_TAG_FEATURE_PATH = params.embedded_tag_feature_path
LOSS_TYPE = params.loss_type

# ベースラインのハイパラ
MODEL_PATH_BASELINE = params.model_path_baseline
EMBEDDED_IMG_FEATURE_PATH_BASELINE = params.embedded_img_feature_path_baseline
EMBEDDED_TAG_FEATURE_PATH_BASELINE = params.embedded_tag_feature_path_baseline

# ディレクトリの作成
SAVE_DIR = f'{BASE_DIR}/layer_loss_same'
os.makedirs(f'{SAVE_DIR}', exist_ok=True)

# ラベル(クラスタID)の取得
img_cluster_id = utils.load_hierarchical_clusterID(IMG_CLUSTER_PATH)
tag_cluster_id = utils.load_hierarchical_clusterID(TAG_CLUSTER_PATH)

# ロスの計算
layer_loss_img2tag, layer_loss_tag2img = \
    calc_layer_loss(EMBEDDED_IMG_FEATURE_PATH, EMBEDDED_TAG_FEATURE_PATH, MODEL_PATH, LOSS_TYPE)
layer_loss_img2tag_baseline, layer_loss_tag2img_baseline = \
    calc_layer_loss(EMBEDDED_IMG_FEATURE_PATH_BASELINE, EMBEDDED_TAG_FEATURE_PATH_BASELINE, MODEL_PATH_BASELINE, LOSS_TYPE)

# ロスのプロット
plt.plot(layer_loss_img2tag.mean(axis=1), label='img2tag_cluster')
plt.plot(layer_loss_tag2img.mean(axis=1), label='tag2img_cluster')
plt.plot(layer_loss_img2tag_baseline.mean(axis=1), label='img2tag_baseline')
plt.plot(layer_loss_tag2img_baseline.mean(axis=1), label='tag2img_baseline')
plt.legend()
plt.savefig(f'{SAVE_DIR}/{DATASET}.png', bbox_inches='tight', dpi=300)
plt.close()