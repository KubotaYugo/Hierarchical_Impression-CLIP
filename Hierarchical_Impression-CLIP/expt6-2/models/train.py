import torch
import numpy as np
import wandb
import json

from HierarchicalDataset import HierarchicalDataset, HierarchicalBatchSampler, HierarchicalDatasetWithoutSampler
import MLP
import ExpMultiplier
import train_utils

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


def main(params):
    with wandb.init(dir=f'{params.expt}/co-embedding'):
        # parameter from config
        config = wandb.config
        EXPT = config.expt
        MAX_EPOCH = config.max_epoch
        EARLY_STOPPING_PATIENCE = config.early_stopping_patience
        LEARNING_RATE = json.loads(config.learning_rate)
        BATCH_SIZE = config.batch_size
        NUM_IMG_CLUSTERS = params.num_img_clusters
        NUM_TAG_CLUSTERS = params.num_tag_clusters
        TAG_PREPROCESS = config.tag_preprocess
        TEMPERATURE = config.temperature
        LEARN_TEMPERATURE = config.learn_temperature
        INITIAL_TEMPERATURE = config.initial_temperature
        LOSS_TYPE = config.loss_type
        CE_BCE = config.ce_bce
        WEIGHTS = json.loads(config.weights)
        RANDOM_SEED = config.random_seed

        # make save directory
        RUN_NAME = f'C=[{NUM_IMG_CLUSTERS}, {NUM_TAG_CLUSTERS}]_{TAG_PREPROCESS}_{TEMPERATURE}_{LEARN_TEMPERATURE}_{INITIAL_TEMPERATURE}_{LOSS_TYPE}_{CE_BCE}_W={WEIGHTS}_seed={RANDOM_SEED}'
        SAVE_DIR = f'{EXPT}/co-embedding/{RUN_NAME}/results'
        os.makedirs(f'{SAVE_DIR}/model', exist_ok=True)

        # set run name
        wandb.run.name = RUN_NAME

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
                
        # set optimizer, criterion, earlystopping
        optimizer = torch.optim.Adam([
            {'params': emb_img.parameters(), 'lr': LEARNING_RATE[0]},
            {'params': emb_tag.parameters(), 'lr': LEARNING_RATE[1]},
            {'params': temperature.parameters(), 'lr': LEARNING_RATE[2]}
            ])
        criterion_CE = torch.nn.CrossEntropyLoss().to(device)
        criterion_BCE = torch.nn.BCEWithLogitsLoss().to(device)
        criterions = [criterion_CE, criterion_BCE]
        earlystopping = train_utils.EarlyStopping(patience=EARLY_STOPPING_PATIENCE, delta=0)

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

            # save results to wandb, csv
            parameters_to_save = {
                    'epoch':                                     epoch,
                    'train_loss_total':                          loss['total'],
                    'train_loss_pair':                           loss['pair'],
                    'train_loss_img':                            loss['img'],
                    'train_loss_tag':                            loss['tag'], 
                    'loss_total_with_temperature_train_all':     loss_with_temperature_train['total'],
                    'loss_pair_with_temperature_train_all':      loss_with_temperature_train['pair'],
                    'loss_img_with_temperature_train_all':       loss_with_temperature_train['img'],
                    'loss_tag_with_temperature_train_all':       loss_with_temperature_train['tag'], 
                    'loss_total_without_temperature_train_all':  loss_without_temperature_train['total'],
                    'loss_pair_without_temperature_train_all':   loss_without_temperature_train['pair'],
                    'loss_img_without_temperature_train_all':    loss_without_temperature_train['img'],
                    'loss_tag_without_temperature_train_all':    loss_without_temperature_train['tag'], 
                    'meanARR_train':                             ARR_train['mean'],
                    'ARR_tag2img_train':                         ARR_train['tag2img'],
                    'ARR_img2tag_train':                         ARR_train['img2tag'],
                    'loss_total_with_temperature_val':           loss_with_temperature_val['total'],
                    'loss_pair_with_temperature_val':            loss_with_temperature_val['pair'],
                    'loss_img_with_temperature_val':             loss_with_temperature_val['img'],
                    'loss_tag_with_temperature_val':             loss_with_temperature_val['tag'], 
                    'loss_total_without_temperature_val':        loss_without_temperature_val['total'],
                    'loss_pair_without_temperature_val':         loss_without_temperature_val['pair'],
                    'loss_img_without_temperature_val':          loss_without_temperature_val['img'],
                    'loss_tag_without_temperature_val':          loss_without_temperature_val['tag'], 
                    'meanARR_val':                               ARR_val['mean'],
                    'ARR_tag2img_val':                           ARR_val['tag2img'],
                    'ARR_img2tag_val':                           ARR_val['img2tag'],
                    'temperture':                                temperature.logit_scale.item()
                    }
            wandb.log(parameters_to_save, step=epoch)
            utils.save_list_to_csv([list(parameters_to_save.values())], f'{SAVE_DIR}/result.csv')

            # early stopping
            earlystopping(ARR_val['mean'])
            if earlystopping.early_stop:
                print('Early stopping')
                train_utils.save_models(f'{SAVE_DIR}/model/checkpoint_{epoch:04d}.pth.tar', emb_img, emb_tag, temperature)
                break

            # save models every 100 epochs
            if epoch%100==0 and epoch!=0:
                train_utils.save_models(f'{SAVE_DIR}/model/checkpoint_{epoch:04d}.pth.tar', emb_img, emb_tag, temperature)
            
            # save models if renew the best meanARR
            if ARR_val['mean']<meanARR_best:
                train_utils.save_models(f'{SAVE_DIR}/model/best.pth.tar', emb_img, emb_tag, temperature)
                meanARR_best = ARR_val['mean']
        
        wandb.log({'meanARR_val_min':meanARR_best}, step=epoch)
        wandb.alert(title='WandBからの通知', text=f'{RUN_NAME}が終了しました．')


if __name__ == '__main__':
    params = utils.get_parameters()
    
    sweep_configuration = {
        'method': 'grid',
        'name': params.expt.replace('/', '_'),
        'metric': {
            'goal': 'minimize',
            'name': 'meanARR_val_min',
        },
        'parameters': {
            'expt': {
                'value': params.expt
            },
            'max_epoch': {
                'value': params.max_epoch
            },
            'early_stopping_patience': {
                'value': params.early_stopping_patience
            },
            'learning_rate': {
                'values': ['[1e-4, 1e-4, 1e-4]']
            },
            'batch_size': {
                'values': [8192]
            },
            'num_img_clusters': {
                'values': [10]
            },
            'num_tag_clusters': {
                'values': [10]
            },
            'tag_preprocess':{
                'values': ['average_single_tag']
            },
            'temperature':{
                'values': ['ExpMultiplierLogit']
            },
            'learn_temperature':{
                'values': [True]
            },
            'initial_temperature':{
                'values': [0.15]
            },
            'loss_type':{
                'values': ['average', 'iterative', 'label_and']
            },
            'ce_bce':{
                'values': ['BCE']
            },
            'random_seed':{
                'values': [1, 2, 3]
            },
            'weights': {
                'values': ['[1.0, 1.0, 1.0]',
                           '[0.5, 1.0, 1.0]',
                           '[1.0, 0.5, 0.5]']
            },
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='Hierarchical_ImpressionCLIP_6-2')
    wandb.agent(sweep_id, function=lambda: main(params))

# python programs/Hierarchical_Impression-CLIP/expt6-2/models/train.py