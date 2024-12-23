import torch
import wandb
import numpy as np

import FontAutoencoder
import FontAutoencoder_utils
import HierarchicalDataset

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


def main(params):
    with wandb.init(dir=f'{params.expt}/FontAutoencoder'):
        # parameter from config
        config = wandb.config
        EXPT = config.expt
        MAX_EPOCH = config.max_epoch
        EARLY_STOPPING_PATIENCE = config.early_stopping_patience
        LEARNING_RATE = config.learning_rate
        BATCH_SIZE = config.batch_size
        DECODER_TYPE = config.decoder_type
        RANDOM_SEED = config.random_seed

        # make save directory
        RUN_NAME = f'decoder={DECODER_TYPE}_batchsize={BATCH_SIZE}_seed={RANDOM_SEED}'
        SAVE_DIR = f'{EXPT}/FontAutoencoder/{RUN_NAME}/results'
        os.makedirs(f'{SAVE_DIR}/model', exist_ok=True)

        # set run name
        wandb.run.name = RUN_NAME

        # fix random numbers, set cudnn option
        utils.fix_seed(RANDOM_SEED)
        
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
        val_loss_min = np.Inf
        for epoch in range(1, MAX_EPOCH + 1):
            print('-'*130)
            print('Epoch {}/{}'.format(epoch, MAX_EPOCH))

            # training and validation
            train_loss = FontAutoencoder_utils.train(trainloader, font_autoencoder, criterion, optimizer, device)
            val_loss = FontAutoencoder_utils.val(valloader, font_autoencoder, criterion, device)
            print(f'[train]  {train_loss:7.6f}')
            print(f'[val]    {val_loss:7.6f}')

            # save results to wandb and csv
            parameters_to_save = {
                    'epoch':      epoch,
                    'train_loss': train_loss,
                    'val_loss':   val_loss,
                    }
            wandb.log(parameters_to_save, step=epoch)
            utils.save_list_to_csv([list(parameters_to_save.values())], f'{SAVE_DIR}/result.csv')

            # early stopping
            earlystopping(val_loss)
            if earlystopping.early_stop:
                filename = f'{SAVE_DIR}/model/checkpoint_{epoch:04d}.pth.tar'
                save_contents = {'font_autoencoder': font_autoencoder.state_dict()}
                torch.save(save_contents, filename)
                break

            # save models if renew the best meanARR
            if val_loss < val_loss_min:
                filename = f'{SAVE_DIR}/model/best.pth.tar'
                save_contents = {'font_autoencoder': font_autoencoder.state_dict()}
                torch.save(save_contents, filename)
                val_loss_min = val_loss

        wandb.log({'val_loss_min':val_loss_min}, step=epoch)
        wandb.alert(title='WandBからの通知', text=f'{RUN_NAME}が終了しました．')


if __name__ == '__main__':
    params = utils.get_parameters()
    
    sweep_configuration = {
        'method': 'grid',
        'name': params.expt.replace('/', '_'),
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss_min',
        },
        'parameters': {
            'expt': {
                'value': params.expt
            },
            'max_epoch': {
                'value': 10000
            },
            'early_stopping_patience': {
                'value': 100
            },
            'learning_rate': {
                'values': [1e-4]
            },
            'batch_size': {
                'values': [16]
            },
            'decoder_type':{
                'values': ['conv', 'deconv']
            },
            'random_seed':{
                'values': [1, 2, 3, 4, 5]
            },
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='Hierarchical_ImpressionCLIP_6-3_FontAutoencoder')
    wandb.agent(sweep_id, function=lambda: main(params))

# python programs/Hierarchical_Impression-CLIP/expt6-3/models/FontAutiencider_pretrain.py