'''
'A'の画像をクラスタ別に保存
'''
import numpy as np
import math

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils

from matplotlib.image import imsave

def save_tiled_images(image_list, tile_size=(10, 10), img_shape=(64, 64), output_path='tiled_image'):
    """
    画像リストからタイル状の画像を生成し、保存する。
    100枚ずつのタイル画像を生成し、N>100のときは複数ファイルに分けて保存する。
    
    :param image_list: グレースケール画像のリスト (要素数N)
    :param tile_size: タイルのサイズ (rows, cols)
    :param img_shape: 各画像の形状 (height, width)
    :param output_path: 出力画像の保存パス（インデックスがファイル名に付加される）
    """
    N = len(image_list)
    tile_rows, tile_cols = tile_size
    img_height, img_width = img_shape
    
    # タイルに必要な画像の数（端数が出る場合も考慮）
    total_tiles = tile_rows * tile_cols
    num_batches = math.ceil(N / total_tiles)  # 100枚ずつバッチに分ける
    
    for batch_idx in range(num_batches):
        # バッチの範囲を計算
        start_idx = batch_idx * total_tiles
        end_idx = min(start_idx + total_tiles, N)  # バッチ内で処理する範囲
        
        # 1枚の大きな画像を作成 (全体の形状を決定)
        tiled_image = np.full((tile_rows * img_height, tile_cols * img_width), 255, dtype=np.uint8)  # 真っ白な画像で初期化
        
        # リストから画像を取り出してタイル状に配置
        for idx, img in enumerate(image_list[start_idx:end_idx]):
            row = idx // tile_cols
            col = idx % tile_cols
            tiled_image[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width] = img
        
        # 保存ファイル名にバッチ番号を付加
        output_file = f"{output_path}_{batch_idx + 1}.png"
        
        # 画像の保存（matplotlibを使用）
        imsave(output_file, tiled_image, cmap='gray')
        print(f"Saved: {output_file}")


# define constant
EXP = utils.EXP
DATASET = 'train'
IMG_CLUSTER_PATH = f'{EXP}/clustering/{DATASET}/image_clusters.npz'
SAVE_DIR = f'{EXP}/clustering/cluster_visualization/PCA/{DATASET}'


# パス，ラベル(クラスタID)の取得
img_paths, _ = utils.load_dataset_paths(DATASET)
img_paths = np.asarray(img_paths)
img_cluster_id = np.load(IMG_CLUSTER_PATH)["arr_0"].astype(np.int64)
number_of_clusters = max(img_cluster_id)+1


for i in range(number_of_clusters):
    SAVE_DIR = f"{EXP}/clustering/cluster_images/{DATASET}/cluster{i}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    img_path_cluster_i = img_paths[img_cluster_id==i]
    imgs = [np.load(img_path)["arr_0"][0] for img_path in img_path_cluster_i] 
    save_tiled_images(imgs, tile_size=(10, 10), img_shape=(64, 64), output_path=f'{SAVE_DIR}/imgs')