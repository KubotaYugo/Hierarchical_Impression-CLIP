import torch
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils
from lib import eval_utils
from lib import retrieve_utils

# define constant
BASE_DIR = utils.BASE_DIR

# for DATASET in ['train', 'val', 'test']:
for DATASET in ['val']:
    SAVE_DIR = f'{BASE_DIR}/retrieve/{DATASET}'
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 特徴量の読み込み
    load_dir = f'{BASE_DIR}/features/{DATASET}'
    embedded_img_features = torch.load(f'{load_dir}/embedded_img_features.pth')
    embedded_tag_features = torch.load(f'{load_dir}/embedded_tag_features.pth')
    embedded_single_tag_features = torch.load(f'{load_dir}/embedded_single_tag_features.pth')

    # 各種検索
    img_paths, tag_paths = utils.load_dataset_paths(DATASET)
    tag_list = list(eval_utils.get_tag_list().values())
    retrieve_utils.retrieve_imp_img(embedded_img_features, embedded_tag_features, img_paths, tag_paths, 'imp2img', 'upper', 10, SAVE_DIR)
    retrieve_utils.retrieve_imp_img(embedded_img_features, embedded_tag_features, img_paths, tag_paths, 'imp2img', 'lower', 10, SAVE_DIR)
    retrieve_utils.retrieve_imp_img(embedded_img_features, embedded_tag_features, img_paths, tag_paths, 'img2imp', 'upper', 10, SAVE_DIR)
    retrieve_utils.retrieve_imp_img(embedded_img_features, embedded_tag_features, img_paths, tag_paths, 'img2imp', 'lower', 10, SAVE_DIR)
    retrieve_utils.retrieve_tag2img(embedded_img_features, embedded_single_tag_features, img_paths, tag_paths, tag_list, 'upper', 10, SAVE_DIR)
    retrieve_utils.retrieve_tag2img(embedded_img_features, embedded_single_tag_features, img_paths, tag_paths, tag_list, 'lower', 10, SAVE_DIR)
    retrieve_utils.retrieve_img2tag(embedded_img_features, embedded_single_tag_features, img_paths, tag_paths, tag_list, SAVE_DIR)