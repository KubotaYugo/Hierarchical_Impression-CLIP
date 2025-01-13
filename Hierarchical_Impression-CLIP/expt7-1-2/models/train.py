import torch
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import wandb
import json

from HierarchicalDataset import DatasetForTrain, DatasetForEval
import MLP
import Temperature
import train_utils

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


def main(params):
    with wandb.init(dir=f'{params.expt}/co-embedding'):

        # GPU config
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True 
        torch.backends.cuda.matmul.allow_tf32 = True

        # AMP用のスケーラーを作成
        scaler = GradScaler()

        # parameter from config
        config = wandb.config
        EXPT = config.expt
        MAX_EPOCH = config.max_epoch
        EARLY_STOPPING_PATIENCE = config.early_stopping_patience
        LEARNING_RATE = json.loads(config.learning_rate)
        BATCH_SIZE = config.batch_size
        TAG_PREPROCESS = config.tag_preprocess
        INITIAL_TEMPERATURE = config.initial_temperature
        LOSS_TYPE = config.loss_type
        WEIGHTS = json.loads(config.weights)
        RANDOM_SEED = config.random_seed

        # make save directory
        RUN_NAME = f'{LOSS_TYPE}_W={WEIGHTS}_BS={BATCH_SIZE}_seed={RANDOM_SEED}'
        SAVE_DIR = f'{EXPT}/co-embedding/{RUN_NAME}/results'
        os.makedirs(f'{SAVE_DIR}/model', exist_ok=True)
        os.makedirs(f'{SAVE_DIR}/embedded_img_feature/train', exist_ok=True)
        os.makedirs(f'{SAVE_DIR}/embedded_tag_feature/train', exist_ok=True)
        os.makedirs(f'{SAVE_DIR}/embedded_img_feature/val', exist_ok=True)
        os.makedirs(f'{SAVE_DIR}/embedded_tag_feature/val', exist_ok=True)
        
        # set run name
        wandb.run.name = RUN_NAME

        # fix random numbers
        utils.fix_seed(RANDOM_SEED)

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
        val_loss_best = np.Inf
        for epoch in range(1, MAX_EPOCH + 1):
            print('-'*130)
            print('Epoch {}/{}'.format(epoch, MAX_EPOCH))

            # training and validation
            loss, layer_loss_img2tag, layer_loss_tag2img = \
                train_utils.train(trainloader, emb_img, emb_tag, temperature, WEIGHTS, LOSS_TYPE, optimizer, scaler)
            loss_without_temperature_train, loss_with_temperature_train, ARR_train, embedded_feature_train = \
                train_utils.val(trainloader_eval, emb_img, emb_tag, temperature)
            loss_without_temperature_val, loss_with_temperature_val, ARR_val, embedded_feature_val = \
                train_utils.val(valloader, emb_img, emb_tag, temperature)
    
            # save embedded feature
            if epoch%50==0 or epoch==1:
                train_utils.save_embedded_feature(embedded_feature_train, SAVE_DIR, 'train', epoch)
                train_utils.save_embedded_feature(embedded_feature_val, SAVE_DIR, 'val', epoch)
        
            print(f'[train]                     {train_utils.loss_format_train(loss)}')
            print(f'[train_all w/  temperature] {train_utils.loss_format_eval(loss_with_temperature_train)}')
            print(f'[train_all w/o temperature] {train_utils.loss_format_eval(loss_without_temperature_train)}')
            print(f'[val       w/  temperature] {train_utils.loss_format_eval(loss_with_temperature_val)}')
            print(f'[val       w/o temperature] {train_utils.loss_format_eval(loss_without_temperature_val)}')
            print(f'[train_all]                 {train_utils.ARR_format(ARR_train)}')
            print(f'[val]                       {train_utils.ARR_format(ARR_val)}')
            
            # save results to wandb, csv
            parameters_to_save = {
                    'epoch':                                           epoch,
                    'train_loss_total':                                loss['total'],
                    'train_loss_img2tag':                              loss['img2tag'],
                    'train_loss_tag2img':                              loss['tag2img'], 
                    'loss_pair_with_temperature_train_all':            loss_with_temperature_train['pair'],
                    'loss_pair_img2tag_with_temperature_train_all':    loss_with_temperature_train['pair_img2tag'],
                    'loss_pair_tag2img_with_temperature_train_all':    loss_with_temperature_train['pair_tag2img'], 
                    'loss_pair_without_temperature_train_all':         loss_without_temperature_train['pair'],
                    'loss_pair_img2tag_without_temperature_train_all': loss_without_temperature_train['pair_img2tag'],
                    'loss_pair_tag2img_without_temperature_train_all': loss_without_temperature_train['pair_tag2img'], 
                    'meanARR_train':                                   ARR_train['mean'],
                    'ARR_tag2img_train':                               ARR_train['tag2img'],
                    'ARR_img2tag_train':                               ARR_train['img2tag'],
                    'loss_pair_with_temperature_val':                  loss_with_temperature_val['pair'],
                    'loss_pair_img2tag_with_temperature_val':          loss_with_temperature_val['pair_img2tag'],
                    'loss_pair_tag2img_with_temperature_val':          loss_with_temperature_val['pair_tag2img'], 
                    'loss_pair_without_temperature_val':               loss_without_temperature_val['pair'],
                    'loss_pair_img2tag_without_temperature_val':       loss_without_temperature_val['pair_img2tag'],
                    'loss_pair_tag2img_without_temperature_val':       loss_without_temperature_val['pair_tag2img'], 
                    'meanARR_val':                                     ARR_val['mean'],
                    'ARR_tag2img_val':                                 ARR_val['tag2img'],
                    'ARR_img2tag_val':                                 ARR_val['img2tag'],
                    'temperture_logit_scale':                          temperature.logit_scale.item()
                    }
            parameters_to_save = {**parameters_to_save, **layer_loss_img2tag, **layer_loss_tag2img}
            wandb.log(parameters_to_save, step=epoch)
            utils.save_list_to_csv([list(parameters_to_save.values())], f'{SAVE_DIR}/result.csv')

            # early stopping
            earlystopping(loss_without_temperature_val['pair'])
            if earlystopping.early_stop:
                print('Early stopping')
                train_utils.save_models(f'{SAVE_DIR}/model/checkpoint_{epoch:04d}.pth.tar', emb_img, emb_tag, temperature)
                break

            # save models every 100 epochs
            if epoch%100==0 and epoch!=0:
                train_utils.save_models(f'{SAVE_DIR}/model/checkpoint_{epoch:04d}.pth.tar', emb_img, emb_tag, temperature)
            
            # save models if renew the val_loss_best
            if loss_without_temperature_val['pair']<val_loss_best:
                train_utils.save_models(f'{SAVE_DIR}/model/best.pth.tar', emb_img, emb_tag, temperature)
                val_loss_best = loss_without_temperature_val['pair']
        
        wandb.log({'loss_pair_without_temperature_val_min':val_loss_best}, step=epoch)
        wandb.alert(title='WandBからの通知', text=f'{RUN_NAME}が終了しました．')


if __name__ == '__main__':
    params = utils.get_parameters()
    
    sweep_configuration = {
        'method': 'grid',
        'name': params.expt.replace('/', '_'),
        'metric': {
            'goal': 'minimize',
            'name': 'loss_pair_without_temperature_val_min',
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
            'tag_preprocess':{
                'values': ['average_single_tag']
            },
            'initial_temperature':{
                'values': [0.15]
            },
            'loss_type':{
                'values': ['SupCon']
            },
            'weights': {
                'values': ['[1.0, 1.0]']
            },
            'random_seed':{
                'values': [1]
            },
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='Hierarchical_ImpressionCLIP_7-1-2')
    wandb.agent(sweep_id, function=lambda: main(params))

# python programs/Hierarchical_Impression-CLIP/expt7-1-2/models/train.py