import torch
import numpy as np

from HierarchicalDataset import HierarchicalDataset, HierarchicalBatchSampler, HierarchicalDatasetWithoutSampler
import ExpMultiplier
import MLP
import train_utils

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils

import warnings
warnings.simplefilter('error', RuntimeWarning)


if __name__ == '__main__':
    params = utils.get_parameters()
    EXPT = params.expt
    MAX_EPOCH = params.max_epoch
    EARLY_STOPPING_PATIENCE = params.early_stopping_patience
    LEARNING_RATE = params.learning_rate
    BATCH_SIZE = params.batch_size
    # BATCH_SIZE = 5
    NUM_IMG_CLUSTERS = params.num_img_clusters
    NUM_TAG_CLUSTERS = params.num_tag_clusters
    TAG_PREPROCESS = params.tag_preprocess
    TEMPERATURE = params.temperature
    LEARN_TEMPERATURE = params.learn_temperature
    INITIAL_TEMPERATURE = params.initial_temperature
    LOSS_TYPE = params.loss_type
    CE_BCE = params.ce_bce
    CE_BCE = 'SupCon'
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
    temperature_class = getattr(ExpMultiplier, TEMPERATURE)
    temperature = temperature_class(INITIAL_TEMPERATURE).to(device)
        
    # set optimizer, criterion
    optimizer = torch.optim.Adam([
        {'params': emb_img.parameters(), 'lr': LEARNING_RATE[0]},
        {'params': emb_tag.parameters(), 'lr': LEARNING_RATE[1]},
        {'params': temperature.parameters(), 'lr': LEARNING_RATE[2]}
        ])
    criterion_CE = torch.nn.CrossEntropyLoss().to(device)
    criterion_BCE = torch.nn.BCEWithLogitsLoss().to(device)
    criterions = [criterion_CE, criterion_BCE]

    # set dataloder, sampler for train
    trainset = HierarchicalDataset('train', EXPT, TAG_PREPROCESS, NUM_IMG_CLUSTERS, NUM_TAG_CLUSTERS)
    sampler = HierarchicalBatchSampler(batch_size=BATCH_SIZE, dataset=trainset)
    trainloader = torch.utils.data.DataLoader(trainset, sampler=sampler, shuffle=False, 
                                              num_workers=os.cpu_count(), batch_size=1, pin_memory=True)
    # train dataloder to calcurate ARR
    train_evalset = HierarchicalDatasetWithoutSampler('train', EXPT, TAG_PREPROCESS, NUM_IMG_CLUSTERS, NUM_TAG_CLUSTERS)
    train_evalloader = torch.utils.data.DataLoader(train_evalset, batch_size=BATCH_SIZE, shuffle=False, 
                                                   num_workers=os.cpu_count(), pin_memory=True)
    # validation dataloder
    valset = HierarchicalDatasetWithoutSampler('val', EXPT, TAG_PREPROCESS, NUM_IMG_CLUSTERS, NUM_TAG_CLUSTERS)
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
        sampler.set_epoch(epoch)
        loss = train_utils.train(trainloader, emb_img, emb_tag, temperature, criterions, 
                     WEIGHTS, LOSS_TYPE, CE_BCE, optimizer, epoch)
        loss_without_temperature_train, loss_with_temperature_train, ARR_train = train_utils.val(train_evalloader, 
                                                                                     emb_img, emb_tag, temperature, 
                                                                                     criterions, WEIGHTS, CE_BCE)
        loss_without_temperature_val, loss_with_temperature_val, ARR_val = train_utils.val(valloader, 
                                                                               emb_img, emb_tag, temperature, 
                                                                               criterions, WEIGHTS, CE_BCE)
        print(f'[train]                     {train_utils.loss_format(loss)}')
        print(f'[train_all w/  temperature] {train_utils.loss_format(loss_with_temperature_train)}')
        print(f'[train_all w/o temperature] {train_utils.loss_format(loss_without_temperature_train)}')
        print(f'[val w/  temperature]       {train_utils.loss_format(loss_with_temperature_val)}')
        print(f'[val w/o temperature]       {train_utils.loss_format(loss_without_temperature_val)}')
        print(f'[train_all]                 meanARR: {ARR_train['mean']:7.2f},  ARR_tag2img: {ARR_train['tag2img']:7.2f},  ARR_img2tag: {ARR_train['img2tag']:7.2f}')
        print(f'[val]                       meanARR: {ARR_val['mean']:7.2f},  ARR_tag2img: {ARR_val['tag2img']:7.2f},  ARR_img2tag: {ARR_val['img2tag']:7.2f}')

        # early stopping
        earlystopping(ARR_val['mean'])
        if earlystopping.early_stop:
            print('Early stopping')
            break