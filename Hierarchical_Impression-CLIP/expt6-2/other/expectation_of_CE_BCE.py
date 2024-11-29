'''
画像のクラスタと印象のクラスタそれぞれを正解ラベルとしたときのBCEの期待値を求める
'''
import torch
import numpy as np

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


# クラスタラベルの取得
params = utils.get_parameters()
IMG_CLUSTER_PATH = params.img_cluster_path
TAG_CLUSTER_PATH = params.tag_cluster_path
img_labels = torch.from_numpy(np.load(IMG_CLUSTER_PATH)["arr_0"].astype(int))[:8]
tag_labels = torch.from_numpy(np.load(TAG_CLUSTER_PATH)["arr_0"].astype(int))[:8]
img_labels_transformed = (img_labels.unsqueeze(0)==img_labels.unsqueeze(1)).float()
tag_labels_transformed = (tag_labels.unsqueeze(0)==tag_labels.unsqueeze(1)).float()

# 乱数で仮の尤度を生成
# random_tensor = torch.rand(img_labels_transformed.size())
random_tensor = (2*torch.rand(img_labels_transformed.size())-1)*(1/0.07)
# random_tensor = torch.zeros(img_labels_transformed.size())
# random_tensor = img_labels_transformed

# random_tensor = torch.eye(img_labels_transformed.shape[0], dtype=float)
# random_tensor[random_tensor==0] = -1
# random_tensor = random_tensor/0.07

# CE, BCE計算
criterion_CE = torch.nn.CrossEntropyLoss()
criterion_BCE = torch.nn.BCEWithLogitsLoss()
CE_pair = criterion_CE(random_tensor, torch.arange(img_labels_transformed.shape[0]))
BCE_tag = criterion_BCE(random_tensor, img_labels_transformed)
BCE_img = criterion_BCE(random_tensor, tag_labels_transformed)
print(CE_pair.item(), BCE_img.item(), BCE_tag.item())