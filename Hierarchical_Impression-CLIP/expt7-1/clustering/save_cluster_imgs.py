'''
クラスタIDを名前として画像を保存
'''


import numpy as np
from matplotlib.image import imsave

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


# define constant
params = utils.get_parameters()
EXPT = params.expt
DATASET = params.dataset
IMG_CLUSTER_PATH = params.img_cluster_path

# 保存用ディレクトリの作成
SAVE_DIR = f'{EXPT}/clustering/save_cluster_imgs/{DATASET}'
os.makedirs(SAVE_DIR, exist_ok=True)

# パス，ラベル(クラスタID)の取得
img_paths, _ = utils.load_dataset_paths(DATASET)
img_paths = np.asarray(img_paths)
img_cluster_id_marge = utils.load_hierarchical_clusterID(IMG_CLUSTER_PATH)

# クラスタIDを名前として画像を保存
fontnames = utils.get_fontnames(DATASET)
imgs = [utils.get_image_to_save(img_path) for img_path in img_paths]
for i in range(len(img_cluster_id_marge)):
    imsave(f'{SAVE_DIR}/{img_cluster_id_marge[i]}_{fontnames[i]}.png', imgs[i], cmap='gray')