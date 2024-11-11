import torch

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils
from models import HierarchicalDataset
from models import FontAutoencoder


params = utils.get_parameters()
DATASET = params.dataset
FONTAUTOENCODER_PATH = params.fontautoencoder_path
IMG_FEATURE_PATH = params.img_feature_path

SAVE_DIR = os.path.dirname(IMG_FEATURE_PATH)
os.makedirs(SAVE_DIR, exist_ok=True)

# データの準備
img_paths, _ = utils.load_dataset_paths(DATASET)
dataset = HierarchicalDataset.ImageDataset(img_paths)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8192, shuffle=False, num_workers = os.cpu_count(), pin_memory=True)

# モデルの準備
device = torch.device('cuda:0')
font_autoencoder = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device)
font_autoencoder.load_state_dict(torch.load(FONTAUTOENCODER_PATH))
font_autoencoder.eval()

# 画像特徴の保存
img_feature_list = []
for data in dataloader:
    input_font = data.to(device)
    with torch.no_grad():
        img_feature = font_autoencoder.encoder(input_font)
    img_feature_list.append(img_feature)
stacked_img_feature = torch.cat(img_feature_list, dim=0)
torch.save(stacked_img_feature, IMG_FEATURE_PATH)