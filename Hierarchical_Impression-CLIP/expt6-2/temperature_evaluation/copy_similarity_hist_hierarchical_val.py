'''
evaluation/similarity_hist_hierarchical.pyの結果をコピーする
'''
from PIL import Image

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


# define constant
params = utils.get_parameters()
EXPT = params.expt

# ディレクトリの作成
save_dir = f'{EXPT}/temperature_evaluation/similarity_hist_hierarchical_val'
os.makedirs(save_dir, exist_ok=True)

temperature_list = [0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for i, initial_temperature in enumerate(temperature_list):
    for random_seed in [1, 2, 3, 4, 5]:
        base_dir = f'{EXPT}/co-embedding/C=[{params.num_img_clusters}, {params.num_tag_clusters}]_{params.tag_preprocess}_{params.temperature}_{params.learn_temperature}_{initial_temperature}_{params.loss_type}_{params.ce_bce}_W={params.weights}_seed={random_seed}'
        load_path = f'{base_dir}/similarity_hist_hierarchical/val.png'
        save_path = f'{save_dir}/{i+1}_initial_temperature={initial_temperature}_seed={random_seed}.png'
        if os.path.exists(load_path):
            image = Image.open(load_path)
            image.save(save_path)

