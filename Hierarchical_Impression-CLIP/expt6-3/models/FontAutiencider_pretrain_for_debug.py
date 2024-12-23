import torch
import numpy as np

import FontAutoencoder
import FontAutoencoder_utils
import HierarchicalDataset

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


# parameter from config
MAX_EPOCH = 10
EARLY_STOPPING_PATIENCE = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
DECODER_TYPE = 'deconv'
RANDOM_SEED = 1

# fix random numbers, set cudnn option
utils.fix_seed(RANDOM_SEED)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# initialize the model and criterion, optimizer, earlystopping
device = torch.device('cuda:0')
font_autoencoder = FontAutoencoder.Autoencoder(DECODER_TYPE).to(device)
criterion = torch.nn.L1Loss().to(device)
optimizer = torch.optim.Adam(font_autoencoder.parameters(), lr=LEARNING_RATE)
earlystopping = FontAutoencoder_utils.EarlyStopping(patience=EARLY_STOPPING_PATIENCE, delta=0)

# set up the data loader
img_paths_train, _ = utils.load_dataset_paths('train')
img_paths_val, _ = utils.load_dataset_paths('val')
train_set = HierarchicalDataset.ImageDataset(img_paths_train)
val_set = HierarchicalDataset.ImageDataset(img_paths_val)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers = os.cpu_count(), pin_memory=True)
valloader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers = os.cpu_count(), pin_memory=True)


# train and save results
for epoch in range(1, MAX_EPOCH + 1):
    print('-'*130)
    print('Epoch {}/{}'.format(epoch, MAX_EPOCH))

    # training and validation
    train_loss = FontAutoencoder_utils.train(trainloader, font_autoencoder, criterion, optimizer, device)
    val_loss = FontAutoencoder_utils.val(valloader, font_autoencoder, criterion, device)
    print(f'[train]  {train_loss:7.4f}')
    print(f'[val]    {val_loss:7.4f}')

    # early stopping
    earlystopping(val_loss)
    if earlystopping.early_stop:
        break