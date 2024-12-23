import torch
import numpy as np

from HierarchicalDataset import DatasetForTrain, DatasetForEval
import Temperature
import MLP
import train_utils

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


if __name__ == '__main__':
    params = utils.get_parameters()
    EXPT = params.expt
    MAX_EPOCH = params.max_epoch
    EARLY_STOPPING_PATIENCE = params.early_stopping_patience
    LEARNING_RATE = params.learning_rate
    # BATCH_SIZE = params.batch_size
    BATCH_SIZE = 5
    TAG_PREPROCESS = params.tag_preprocess
    TEMPERATURE = params.temperature
    LEARN_TEMPERATURE = params.learn_temperature
    INITIAL_TEMPERATURE = params.initial_temperature
    # LOSS_TYPE = params.loss_type
    LOSS_TYPE = ['SupCon', 'BCE'][1]
    WEIGHTS = params.weights
    RANDOM_SEED = params.random_seed

    # fix random numbers, set cudnn option
    utils.fix_seed(RANDOM_SEED)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # set MLPs and learnable temperture parameter
    device = torch.device('cuda:0')
    emb_img = MLP.ReLU().to(device)
    emb_tag = MLP.ReLU().to(device)
    temperature = Temperature.Temperature(INITIAL_TEMPERATURE).to(device)
        
    # set optimizer
    optimizer = torch.optim.Adam([
        {'params': emb_img.parameters(),     'lr': LEARNING_RATE[0]},
        {'params': emb_tag.parameters(),     'lr': LEARNING_RATE[1]},
        {'params': temperature.parameters(), 'lr': LEARNING_RATE[2]}
        ])

    # initializing the data loader
    # train
    trainset = DatasetForTrain('train', EXPT, TAG_PREPROCESS)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, 
                                              num_workers=os.cpu_count(), pin_memory=True, drop_last=True)
    # train (eval)
    trainset_eval = DatasetForEval('train', EXPT, TAG_PREPROCESS)
    trainloader_eval = torch.utils.data.DataLoader(trainset_eval, batch_size=BATCH_SIZE, shuffle=False, 
                                                   num_workers=os.cpu_count(), pin_memory=True)
    # val
    valset = DatasetForEval('val', EXPT, TAG_PREPROCESS)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, 
                                            num_workers=os.cpu_count(), pin_memory=True)

    # early stopping
    earlystopping = train_utils.EarlyStopping(patience=EARLY_STOPPING_PATIENCE, delta=0)

    # train and save results
    meanARR_best = np.Inf
    for epoch in range(1, MAX_EPOCH + 1):
        print('-'*130)
        print('Epoch {}/{}'.format(epoch, MAX_EPOCH))

        # training and validation
        loss, layer_loss_img2tag, layer_loss_tag2img = \
            train_utils.train(trainloader, emb_img, emb_tag, temperature, WEIGHTS, LOSS_TYPE, optimizer)
        loss_without_temperature_train, loss_with_temperature_train, ARR_train = \
            train_utils.val(trainloader_eval, emb_img, emb_tag, temperature)
        loss_without_temperature_val, loss_with_temperature_val, ARR_val = \
            train_utils.val(valloader, emb_img, emb_tag, temperature)
        
        print(f'[train]                     {train_utils.loss_format_train(loss)}')
        print(f'[train_all w/  temperature] {train_utils.loss_format_eval(loss_with_temperature_train)}')
        print(f'[train_all w/o temperature] {train_utils.loss_format_eval(loss_without_temperature_train)}')
        print(f'[val       w/  temperature] {train_utils.loss_format_eval(loss_with_temperature_val)}')
        print(f'[val       w/o temperature] {train_utils.loss_format_eval(loss_without_temperature_val)}')
        print(f'[train_all]                 {train_utils.ARR_format(ARR_train)}')
        print(f'[val]                       {train_utils.ARR_format(ARR_val)}')

        # early stopping
        earlystopping(ARR_val['mean'])
        if earlystopping.early_stop:
            print('Early stopping')
            break