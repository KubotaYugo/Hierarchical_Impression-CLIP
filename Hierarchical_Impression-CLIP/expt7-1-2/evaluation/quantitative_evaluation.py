import torch
import numpy as np

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils
from lib import eval_utils
# import models.Temperature as Temperature
# from models.HierarchicalClipLoss import calc_hierarchical_clip_loss_eval

# define constant
params = utils.get_parameters()
DATASET = params.dataset
BASE_DIR = params.base_dir
IMG_CLUSTER_PATH = params.img_cluster_path
TAG_CLUSTER_PATH = params.tag_cluster_path
EMBEDDED_IMG_FEATURE_PATH = params.embedded_img_feature_path
EMBEDDED_TAG_FEATURE_PATH = params.embedded_tag_feature_path
EMBEDDED_SINGLE_TAG_FEATURE_PATH = params.embedded_single_tag_feature_path

# ロス計算用
MODEL_PATH = params.model_path
TEMPERATURE = params.temperature
WEIGHTS = params.weights
CE_BCE = params.ce_bce


# 特徴量の読み込み
embedded_img_feature = torch.load(EMBEDDED_IMG_FEATURE_PATH)
embedded_tag_feature = torch.load(EMBEDDED_TAG_FEATURE_PATH)
embedded_single_tag_feature = torch.load(EMBEDDED_SINGLE_TAG_FEATURE_PATH)

# Retrieval Rankの計算
similarity_matrix = torch.matmul(embedded_img_feature, embedded_tag_feature.T)
RR_tag2img = eval_utils.retrieval_rank(similarity_matrix, 'tag2img')
RR_img2tag = eval_utils.retrieval_rank(similarity_matrix, 'img2tag')

# mean Average Precisionの計算
img_paths, tag_paths = utils.load_dataset_paths(DATASET)
tag_list = list(utils.get_tag_list())
AP_tag2img = eval_utils.AP_tag2img(embedded_img_feature, embedded_single_tag_feature, tag_list, tag_paths)
AP_img2tag = eval_utils.AP_img2tag(embedded_img_feature, embedded_single_tag_feature, tag_list, tag_paths)

# # loss計算用のラベルの作成
# device = torch.device('cuda:0')
# img_cluster_id = torch.from_numpy(np.load(IMG_CLUSTER_PATH)['arr_0'].astype(np.int64)).to(device)
# tag_cluster_id = torch.from_numpy(np.load(TAG_CLUSTER_PATH)['arr_0'].astype(np.int64)).to(device)
# img_labels_transformed = (img_cluster_id.unsqueeze(0)==img_cluster_id.unsqueeze(1)).float()
# tag_labels_transformed = (tag_cluster_id.unsqueeze(0)==tag_cluster_id.unsqueeze(1)).float()
# pair_labels = torch.arange(img_cluster_id.shape[0]).to(device)
# labels = [pair_labels, img_labels_transformed, tag_labels_transformed]
# # 温度パラメータの読み込み
# temperature_class = getattr(ExpMultiplier, TEMPERATURE)
# temperature = temperature_class().to(device)
# model_params = torch.load(MODEL_PATH)
# temperature.load_state_dict(model_params['temperature'])
# # 評価指標の準備
# criterion_CE = torch.nn.CrossEntropyLoss().to(device)
# criterion_BCE = torch.nn.BCEWithLogitsLoss().to(device)
# criterions = [criterion_CE, criterion_BCE]
# # ロス計算
# loss_without_temperature, loss_with_temperature = \
# calc_hierarchical_clip_loss_eval(embedded_img_feature, embedded_tag_feature, temperature, WEIGHTS, criterions, labels, CE_BCE)

# 結果の表示
# print('----------ロス(温度なし)----------')
# print(f'loss_total: {loss_without_temperature['total']:.4f}')
# print(f'loss_pair:  {loss_without_temperature['pair']:.4f}')
# print(f'loss_img:   {loss_without_temperature['img']:.4f}')
# print(f'loss_tag:   {loss_without_temperature['tag']:.4f}')

# print('----------ロス(温度あり)----------')
# print(f'loss_total: {loss_with_temperature['total']:.4f}')
# print(f'loss_pair:  {loss_with_temperature['pair']:.4f}')
# print(f'loss_img:   {loss_with_temperature['img']:.4f}')
# print(f'loss_tag:   {loss_with_temperature['tag']:.4f}')

print('----------検索精度----------')
print(f'meanARR:     {(np.mean(RR_tag2img)+np.mean(RR_img2tag))/2:.2f}')
print(f'ARR_tag2img: {np.mean(RR_tag2img):.2f}')
print(f'ARR_img2tag: {np.mean(RR_img2tag):.2f}')
print(f'AP_tag2img:  {np.mean(AP_tag2img):.4f}')
print(f'AP_img2tag:  {np.mean(AP_img2tag):.4f}')