import torch

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils
from models import HierarchicalDataset
import FontAutoencoder


# set the parameters
DATASET = 'train'
BATCH_SIZE = 16
DECODER_TYPE = 'deconv'
RANDOM_SEED = 1

# fix random numbers, set cudnn option
utils.fix_seed(RANDOM_SEED)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False      # Disable dynamic optimizations for reproducibility
torch.backends.cudnn.deterministic = True   # Ensure deterministic behavior

# initialize the model and criterion, optimizer, earlystopping
device = torch.device('cuda:0')
font_autoencoder = FontAutoencoder.Autoencoder(DECODER_TYPE).to(device)
font_autoencoder.eval()

# set up the data loader
img_paths, _ = utils.load_dataset_paths(DATASET)
dataset = HierarchicalDataset.ImageDataset(img_paths)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = os.cpu_count(), pin_memory=True)

# reconstruct images
reconstructed_img_list = []
with torch.no_grad():
    for data in dataloader:
        # Forward pass and loss computation
        input_imgs = data.to(device)
        output_imgs = font_autoencoder(input_imgs)
        reconstructed_img_list.append(output_imgs)
stacked_reconstructed_img = torch.cat(reconstructed_img_list, dim=0)
stacked_reconstructed_img = stacked_reconstructed_img.to('cpu').detach().numpy().copy()

# save imgs
# 入力画像と再構成画像を上下に並べて保存
for i, img in enumerate(stacked_reconstructed_img):
        images = img[0]
        for c in range(1,26):
            images = np.hstack([images, pad_h, img[c]])    