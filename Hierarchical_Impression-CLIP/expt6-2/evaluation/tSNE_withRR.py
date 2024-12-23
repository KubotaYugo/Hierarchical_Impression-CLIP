'''
Retrieval Rankの大小などで色付けして画像特徴と印象特徴を別々にプロット
'''
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import torch
import numpy as np
import matplotlib.pyplot as plt
from lib import utils
from lib import eval_utils


def plot(x, y, color, name):
    fig, ax = plt.subplots(figsize=(6.4*1.5, 4.8*1.5))
    sort_indices = np.argsort(np.abs(color-np.mean(color)))
    x = x[sort_indices]
    y = y[sort_indices]
    color = np.asarray(color)
    color = color[sort_indices]
    scatter = plt.scatter(x, y, c=color, alpha=0.8, edgecolors='w', linewidths=0.1, s=5, cmap='jet')
    plt.colorbar(scatter)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.savefig(f"{SAVE_DIR}/{name}.png", bbox_inches='tight', dpi=500)
    plt.close()


# define constant
params = utils.get_parameters()
DATASET = params.dataset
BASE_DIR = params.base_dir
EMBEDDED_IMG_FEATURE_PATH = params.embedded_img_feature_path
EMBEDDED_TAG_FEATURE_PATH = params.embedded_tag_feature_path

# 保存用フォルダの準備
SAVE_DIR = f"{BASE_DIR}/tSNE_withRR/{DATASET}"
os.makedirs(f"{SAVE_DIR}", exist_ok=True)

# 特徴量の読み込み
embedded_img_feature = torch.load(EMBEDDED_IMG_FEATURE_PATH)
embedded_tag_feature = torch.load(EMBEDDED_TAG_FEATURE_PATH)

# Retrieval Rankの計算
similarity_matrix = torch.matmul(embedded_img_feature, embedded_tag_feature.T)
RR_img2tag = eval_utils.retrieval_rank(similarity_matrix, "img2tag")
RR_tag2img = eval_utils.retrieval_rank(similarity_matrix, "tag2img")

# tSNE特徴量の読み込み
tSNE_feature_filename = f'{BASE_DIR}/tSNE/{DATASET}/tSNE_feature.npz'
tSNE_embedding = np.load(tSNE_feature_filename)['arr_0']

# RR_img2tag, RR_tag2imgで色付けした，画像特徴，印象特徴の分布をプロット
x = tSNE_embedding[:, 0]
y = tSNE_embedding[:, 1]
image_x = x[:len(embedded_img_feature)]
image_y = y[:len(embedded_img_feature)]
tag_x = x[len(embedded_img_feature):]
tag_y = y[len(embedded_img_feature):]
x_min = min(x)-5
x_max = max(x)+5
y_min = min(y)-5
y_max = max(y)+5
plot(image_x, image_y, RR_img2tag, 'img2tag_img')
plot(image_x, image_y, RR_tag2img, 'tag2img_img')
plot(tag_x, tag_y, RR_img2tag, 'img2tag_tag')
plot(tag_x, tag_y, RR_tag2img, 'tag2img_tag')

# RR_img2tag, RR_tag2imgの差分で色付けした，画像特徴，印象特徴の分布をプロット
RR_img2tag = np.asarray(RR_img2tag)
RR_tag2img = np.asarray(RR_tag2img)
plot(image_x, image_y, RR_img2tag-RR_tag2img, 'RR_img2tag-RR_tag2img_img')
plot(tag_x, tag_y, RR_tag2img-RR_img2tag, 'RR_tag2img-RR_img2tag_tag')

# インデックスで色付けした画像特徴，印象特徴の分布をプロット
img_distance_from_top_left = np.sqrt(pow(-80-image_x,2)+pow(80-image_y,2))
tag_distance_from_top_left = np.sqrt(pow(-80-tag_x,2)+pow(80-tag_y,2))

plot(image_x, image_y, image_x, 'index_img_x_img')
plot(tag_x, tag_y, image_x, 'index_img_x_tag')
plot(image_x, image_y, tag_x, 'index_tag_x_img')
plot(tag_x, tag_y, tag_x, 'index_tag_x_tag')

plot(image_x, image_y, image_y, 'index_img_y_img')
plot(tag_x, tag_y, image_y, 'index_img_y_tag')
plot(image_x, image_y, tag_y, 'index_tag_y_img')
plot(tag_x, tag_y, tag_y, 'index_tag_y_tag')

plot(image_x, image_y, img_distance_from_top_left, 'index_img_c_img')
plot(tag_x, tag_y, img_distance_from_top_left, 'index_img_c_tag')
plot(image_x, image_y, tag_distance_from_top_left, 'index_tag_c_img')
plot(tag_x, tag_y, tag_distance_from_top_left, 'index_tag_c_tag')