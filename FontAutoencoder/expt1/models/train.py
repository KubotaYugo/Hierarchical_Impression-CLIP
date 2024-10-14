import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
import csv
import wandb

import autoencoder

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


class EarlyStopping:
    def __init__(self, patience, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.early_stop = False
        self.min_value = np.Inf
    def __call__(self, value):
        if self.min_value+self.delta <= value:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            if  value < self.min_value:
                print(f'Validation loss decreased ({self.min_value:.4f} --> {value:.4f})')
                self.min_value = value


def train(dataloader, font_autoencoder, criterion, optimizer):
    # one epoch training
    font_autoencoder.train()

    # Iterate over data
    loss_list = []
    data_num = []
    for idx, data in enumerate(dataloader):
        inputs = data.cuda(non_blocking=True)
        # forward
        with torch.set_grad_enabled(True):
            # get model outputs and calculate loss
            outputs = font_autoencoder(inputs)
            loss = criterion(outputs, inputs)
            data_num.append(inputs.shape[0])
            loss_list.append(loss.item())
            # backward + optimize only if in training phase
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    total_batch_loss = (np.asarray(loss_list)*np.asarray(data_num)).sum()/np.asarray(data_num).sum()
    return total_batch_loss


def val(dataloader, font_autoencoder, criterion):
    # one epoch training
    font_autoencoder.train()

    # Iterate over data
    loss_list = []
    data_num = []
    for idx, data in enumerate(dataloader):
        inputs = data.cuda(non_blocking=True)
        # forward
        with torch.no_grad():
            # get model outputs and calculate loss
            outputs = font_autoencoder(inputs)
            loss = criterion(outputs, inputs)
            data_num.append(inputs.shape[0])
            loss_list.append(loss.item())
        
    total_batch_loss = (np.asarray(loss_list)*np.asarray(data_num)).sum()/np.asarray(data_num).sum()
    return total_batch_loss


def main(config=None):
    with wandb.init(config=config):
        # parameter from config
        config = wandb.config
        LR = config.learning_rate
        BATCH_SIZE = config.batch_size
        EXP = config.expt
        MAX_EPOCH = config.max_epoch
        EARLY_STOPPING_PATIENCE = config.early_stopping_patience
   
        # make save directory
        SAVE_FOLDER = f'{EXP}/LR={LR}, BS={BATCH_SIZE}/results'
        save_folder = SAVE_FOLDER + '/model'
        os.makedirs(save_folder, exist_ok=True)

        # set run name
        wandb.run.name = f'LR={LR}, BS={BATCH_SIZE}'

        # get args, fix random numbers, set cudnn option
        utils.fix_seed(7)
        cudnn.enabled = True
        cudnn.benchmark = True

        # model, criterion, optimizer
        device = torch.device('cuda:0')
        font_autoencoder = autoencoder.autoencoder().to(device)
        criterion = nn.L1Loss().to(device)
        optimizer = optim.Adam(font_autoencoder.parameters(), lr=LR)
        
        # dataloder
        font_paths_train = utils.get_font_paths("train")
        font_paths_val = utils.get_font_paths("val")
        train_set = utils.ImageDataset(font_paths_train)
        val_set = utils.ImageDataset(font_paths_val)
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers = os.cpu_count(), pin_memory=True)
        valloader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers = os.cpu_count(), pin_memory=True)

        # early stopping
        earlystopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, delta=0)


        # train and save results
        val_loss_best = np.Inf
        for epoch in range(1, MAX_EPOCH + 1):
            print('Epoch {}/{}'.format(epoch, MAX_EPOCH))
            print('-'*10)

            # training and validation
            train_loss = train(trainloader, font_autoencoder, criterion, optimizer)
            val_loss = val(valloader, font_autoencoder, criterion)
            print(f"[train] loss: {train_loss:.4f}")
            print(f"[val]   loss: {val_loss:.4f}")
                    
            # save results to wandb
            wandb.log({"loss_train": train_loss, "loss_val": val_loss}, step = epoch)

            # save results to csv file
            f_epoch = open(f"{SAVE_FOLDER}/result_epoch.csv", 'a')
            csv_writer = csv.writer(f_epoch)
            csv_writer.writerow([epoch, train_loss, val_loss])
            f_epoch.close()
            
            # early stopping
            earlystopping(val_loss)
            if earlystopping.early_stop:
                print("Early stopping")
                break

            # save models and optimizer every 100 epochs
            if epoch%100==0 and epoch!=0:
                filename = f'{SAVE_FOLDER}/model/checkpoint_{epoch:04d}.pth.tar'
                save_contents = {'font_autoencoder':font_autoencoder.state_dict(), 'optimizer':optimizer.state_dict()}
                torch.save(save_contents, filename)
            
            # save models and optimizer if renew the best val_loss
            if val_loss < val_loss_best:
                filename = f"{SAVE_FOLDER}/model/best.pth.tar"
                save_contents = {'font_autoencoder':font_autoencoder.state_dict(), 'optimizer':optimizer.state_dict()}
                torch.save(save_contents, filename)
                val_loss_best = val_loss

        wandb.alert(
        title="WandBからの通知", 
        text=f"LR={LR}, BS={BATCH_SIZE}が終了しました．"
        )


if __name__ == '__main__':

    expt = utils.EXPT
    max_epoch = utils.MAX_EPOCH
    early_stopping_patience = utils.EARLY_STOPPING_PATIENCE

    sweep_configuration = {
    'method': 'grid',  # 'random'や'grid'を指定可能
    'name': expt.replace('/', '_'),
    'metric': {
        'goal': 'minimize',
        'name': 'loss_val',
    },
    'parameters': {
        'batch_size': {
            'values': [8, 16, 32, 64, 256, 1028]
        },
        'learning_rate': {
            'values': [1e-3, 1e-4]
        },
        'max_epoch': {
            'value': max_epoch
        },
        'early_stopping_patience': {
            'value': early_stopping_patience
        },
        'expt': {
            'value': expt
        }
    }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='autoencoder')
    wandb.agent(sweep_id, main)