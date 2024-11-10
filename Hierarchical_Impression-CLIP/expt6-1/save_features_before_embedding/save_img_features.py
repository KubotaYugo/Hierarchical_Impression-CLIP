import torch

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils
from models import HierarchicalDataset
from models import FontAutoencoder

params = utils.get_parameters()
EXPT = params['expt']
DATASET = params['dataset']
FONTAUTOENCODER_PATH = params['fontautoencoder_path']
BATCH_SIZE = params['batch_size']
IMG_FEATURE_PATH = params['img_feature_path']

SAVE_DIR = os.path.dirname(IMG_FEATURE_PATH)
os.makedirs(SAVE_DIR, exist_ok=True)

# データの準備
img_paths, tag_paths = utils.load_dataset_paths(DATASET)
dataset = HierarchicalDataset.ImageDataset(img_paths)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = os.cpu_count(), pin_memory=True)

# モデルの準備
device = torch.device('cuda:0')
font_autoencoder = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device)
font_autoencoder.load_state_dict(torch.load(FONTAUTOENCODER_PATH))
font_autoencoder.eval()

# 画像特徴の保存
with torch.no_grad():
    for i, data in enumerate(dataloader):
        input_font = data.to(device)
        font_feature = font_autoencoder.encoder(input_font)
        if i==0:
            stacked_font_features = font_feature
        else:
            stacked_font_features = torch.concatenate((stacked_font_features, font_feature), dim=0)
    torch.save(stacked_font_features, IMG_FEATURE_PATH)