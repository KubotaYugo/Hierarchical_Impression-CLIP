'''
'A'の画像を画像クラスタ/印象クラスタ別に保存
'''
import numpy as np
import math
from matplotlib.image import imsave

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


def save_tiled_images(image_list=None, tile_size=(10, 10), img_shape=(64, 64), filename=None):
    '''
    画像リストからタイル状の画像を生成し、保存する。
    100枚ずつのタイル画像を生成し、N>100のときは複数ファイルに分けて保存する。

    :param image_list: グレースケール画像のリスト (要素数N)
    :param tile_size: タイルのサイズ (rows, cols)
    :param img_shape: 各画像の形状 (height, width)
    :param output_path: 出力画像の保存パス（インデックスがファイル名に付加される）
    '''
    N = len(image_list)
    tile_rows, tile_cols = tile_size
    img_height, img_width = img_shape
    
    # タイルに必要な画像の数（端数が出る場合も考慮）
    total_tiles = tile_rows * tile_cols
    num_batches = math.ceil(N / total_tiles)
    
    for batch_idx in range(num_batches):
        # バッチの範囲を計算
        start_idx = batch_idx * total_tiles
        end_idx = min(start_idx + total_tiles, N)
        # 1枚の大きな画像を作成 (全体の形状を決定)
        tiled_image = np.full((tile_rows * img_height, tile_cols * img_width), 255, dtype=np.uint8) 
        # リストから画像を取り出してタイル状に配置
        for idx, img in enumerate(image_list[start_idx:end_idx]):
            row = idx // tile_cols
            col = idx % tile_cols
            tiled_image[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width] = img
        # 画像の保存
        output_file = f'{filename}_{batch_idx + 1}.png'
        imsave(output_file, tiled_image, cmap='gray')
        print(f'Saved: {output_file}')


# define constant
params = utils.get_parameters()
EXPT = params.expt
DATASET = params.dataset
IMG_CLUSTER_PATH = params.img_cluster_path
TAG_CLUSTER_PATH = params.tag_cluster_path
NUM_IMG_CLUSTERS = params.num_img_clusters
NUM_TAG_CLUSTERS = params.num_tag_clusters
BASE_SAVE_DIR = f'{EXPT}/clustering/cluster_imgs/num_img_cluster={NUM_IMG_CLUSTERS}_num_tag_cluster={NUM_TAG_CLUSTERS}'

# パス，ラベル(クラスタID)の取得
img_paths, _ = utils.load_dataset_paths(DATASET)
img_paths = np.asarray(img_paths)
img_cluster_id = np.load(IMG_CLUSTER_PATH)['arr_0'].astype(np.int64)
tag_cluster_id = np.load(TAG_CLUSTER_PATH)['arr_0'].astype(np.int64)

# 画像クラスタ別に画像を保存
SAVE_DIR = f'{BASE_SAVE_DIR}/img_cluster/{DATASET}'
os.makedirs(SAVE_DIR, exist_ok=True)
for i in range(NUM_IMG_CLUSTERS):
    for j in range(NUM_TAG_CLUSTERS):
        mask = (img_cluster_id==i)*(tag_cluster_id==j)
        img_path_cluster_ij = img_paths[mask]
        imgs = [np.load(img_path)['arr_0'][0] for img_path in img_path_cluster_ij]
        save_tiled_images(imgs, filename=f'{SAVE_DIR}/[img_cluster, tag_cluster]=[{i+1}, {j+1}]')

# 印象クラスタ別に画像を保存
SAVE_DIR = f'{BASE_SAVE_DIR}/tag_cluster/{DATASET}'
os.makedirs(SAVE_DIR, exist_ok=True)
for i in range(NUM_TAG_CLUSTERS):
    for j in range(NUM_IMG_CLUSTERS):
        mask = (tag_cluster_id==i)*(img_cluster_id==j)
        img_path_cluster_ij = img_paths[mask]
        imgs = [np.load(img_path)['arr_0'][0] for img_path in img_path_cluster_ij]
        save_tiled_images(imgs, filename=f'{SAVE_DIR}/[tag_cluster, img_cluster]=[{i+1}, {j+1}]')