import torch
from torch.utils.data import Dataset
import random
import numpy as np
import csv

# define constant
EXPT = 'autoencoder/expt1/pretrain_deconv'
MAX_EPOCH = 10000
EARLY_STOPPING_PATIENCE = 100


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm


def get_font_paths(dataset):
    with open(f"dataset/MyFonts_preprocessed/tag_txt/fontname/{dataset}.csv") as f:
        reader = csv.reader(f)
        font_names = np.asarray([row for row in reader])
    font_paths = [f"dataset/MyFonts_preprocessed/font_numpy_Impression-CLIP/{dataset}/{font_name[0]}.npz" for font_name in font_names]
    return font_paths


class ImageDataset(Dataset):
    def __init__(self, font_paths):
        self.font_paths = font_paths
    def __len__(self):
        return len(self.font_paths)
    def __getitem__(self, idx):
        font = np.load(self.font_paths[idx])["arr_0"].astype(np.float32)
        font = torch.from_numpy(font/255)   #0~255で保存しているので，0~1に
        return font