from transformers import CLIPTokenizer, CLIPModel
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
import csv
import wandb
import json

from HierarchicalDataset import HierarchicalDataset, HierarchicalBatchSampler
from HierarchicalClipLoss import calc_hierarchical_clip_loss, calc_loss_pair
import FontAutoencoder
import MLP
import ExpMultiplier

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils
from lib import eval_utils

import warnings
warnings.simplefilter("error", RuntimeWarning)


class EarlyStopping:
    '''
    meanARRがpatience回以上更新されなければself.early_stopをTrueに
    '''
    def __init__(self, patience, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.early_stop = False
        self.min_value = np.Inf
    def __call__(self, value):
        if value >= self.min_value-self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            print(f'Validation meanARR decreased ({self.min_value} --> {value})')
            self.min_value = value


def train(dataloader, models, temperature, criterions, weights, optimizer):
    # one epoch training
    font_autoencoder, clip_model, emb_i, emb_t = models
    font_autoencoder.eval()
    clip_model.eval()
    emb_i.train()
    emb_t.train()

    # Iterate over data
    loss_total_list = []
    loss_pair_list = []
    loss_img_list = []
    loss_tag_list = []
    for idx, data in enumerate(dataloader):
        imgs, tokenized_tags, img_labels, tag_labels = data
        imgs, tokenized_tags, img_labels, tag_labels = imgs[0], tokenized_tags[0], img_labels[0], tag_labels[0]
        imgs = imgs.cuda(non_blocking=True)
        tokenized_tags = tokenized_tags.cuda(non_blocking=True)
        img_labels = img_labels[:,0].cuda(non_blocking=True)
        tag_labels = tag_labels[:,0].cuda(non_blocking=True)

        # prepare labels 
        pair_labels = torch.arange(imgs.shape[0]).to('cuda')
        img_labels_transformed = (img_labels.unsqueeze(0) == img_labels.unsqueeze(1)).float()
        tag_labels_transformed = (tag_labels.unsqueeze(0) == tag_labels.unsqueeze(1)).float()
        labels = [pair_labels, img_labels_transformed, tag_labels_transformed]

        # forward
        with torch.no_grad():
            img_features = font_autoencoder.encoder(imgs)
            tag_features = clip_model.get_text_features(tokenized_tags)
        with torch.set_grad_enabled(True):
            # get model outputs
            embedded_img_features = emb_i(img_features)
            embedded_tag_features = emb_t(tag_features)
            loss_total, loss_pair, loss_img, loss_tag = calc_hierarchical_clip_loss(embedded_img_features, embedded_tag_features,
                                                                                    temperature, weights, criterions, labels)
            # append loss to list
            loss_total_list.append(loss_total.item())
            loss_pair_list.append(loss_pair.item())
            loss_img_list.append(loss_img.item())
            loss_tag_list.append(loss_tag.item())

            # backward and optimize
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
    
    aveloss_total = np.mean(loss_total_list)
    aveloss_pair = np.mean(loss_pair_list)
    aveloss_img = np.mean(loss_img_list)
    aveloss_tag = np.mean(loss_tag_list)
    loss_dict = {'total':aveloss_total, 'pair':aveloss_pair, 'img':aveloss_img, 'tag':aveloss_tag}

    return loss_dict


def val(dataloader, models, temperature, criterion_CE):
    # one epoch validation
    font_autoencoder, clip_model, emb_i, emb_t = models
    font_autoencoder.eval()
    clip_model.eval()
    emb_i.eval()
    emb_t.eval()

    with torch.no_grad():
        # extruct embedded features
        for idx, data in enumerate(dataloader):
            imgs, tokenized_tags = data
            imgs = imgs.cuda(non_blocking=True)
            tokenized_tags = tokenized_tags.cuda(non_blocking=True)

            # forward
            img_features = font_autoencoder.encoder(imgs)
            tag_features = clip_model.get_text_features(tokenized_tags) 
            embedded_img_features = emb_i(img_features)
            embedded_tag_features = emb_t(tag_features)
            
            if idx==0:
                embedded_img_features_stack = embedded_img_features
                embedded_tag_features_stack = embedded_tag_features
            else:
                embedded_img_features_stack = torch.concatenate((embedded_img_features_stack, embedded_img_features), dim=0)
                embedded_tag_features_stack = torch.concatenate((embedded_tag_features_stack, embedded_tag_features), dim=0)
        
        
        loss_pair = calc_loss_pair(embedded_img_features_stack, embedded_tag_features_stack, temperature, criterion_CE)
        
        # culculate Average Retrieval Rank
        similarity_matrix = torch.matmul(embedded_img_features_stack, embedded_tag_features_stack.T)
        ARR_tag2img = np.mean(eval_utils.retrieval_rank(similarity_matrix, "tag2img"))
        ARR_img2tag = np.mean(eval_utils.retrieval_rank(similarity_matrix, "img2tag"))
        meanARR = (ARR_img2tag+ARR_tag2img)/2
        ARR_dict = {'mean':meanARR, 'tag2img':ARR_tag2img, 'img2tag':ARR_img2tag}
        
    return ARR_dict, loss_pair.item()


def main(config=None):
    with wandb.init(config=config, dir=exp):
        # parameter from config
        config = wandb.config
        EXP = config.expt
        MAX_EPOCH = config.max_epoch
        EARLY_STOPPING_PATIENCE = config.early_stopping_patience
        LR = config.lr
        BATCH_SIZE = config.batch_size
        WEIGHTS = json.loads(config.weights)
        INITIAL_TEMPERATURE = config.initial_temperature

        FONT_AUTOENCODER_PATH = utils.FONT_AUTOENCODER_PATH
        IMG_CLUSTER_PATH = utils.IMG_CLUSTER_PATH
        TAG_CLUSTER_PATH = utils.TAG_CLUSTER_PATH

        # make save directory
        SAVE_DIR = f'{EXP}/LR={LR}_BS={BATCH_SIZE}_W={WEIGHTS}/results'
        os.makedirs(f'{SAVE_DIR}/model', exist_ok=True)

        # set run name and wandb dir
        wandb.run.name = f'LR={LR}_BS={BATCH_SIZE}_W={WEIGHTS}'

        # fix random numbers, set cudnn option
        utils.fix_seed(7)
        cudnn.enabled = True
        cudnn.benchmark = True

        # set model and optimized parameters
        device = torch.device('cuda:0')
        font_autoencoder = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device)
        font_autoencoder.load_state_dict(torch.load(FONT_AUTOENCODER_PATH))
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        emb_i = MLP.ReLU().to(device)
        emb_t = MLP.ReLU().to(device)
        temperature = ExpMultiplier.ExpMultiplier(INITIAL_TEMPERATURE).to(device)
        models = [font_autoencoder, clip_model, emb_i, emb_t]
            
        # set optimizer, criterion
        optimizer = torch.optim.Adam(list(emb_i.parameters())+list(emb_t.parameters())+list(temperature.parameters()), lr=LR)
        criterion_CE = nn.CrossEntropyLoss().to(device)
        criterion_BCE = nn.BCEWithLogitsLoss().to(device)
        criterions = [criterion_CE, criterion_BCE]

        # set dataloder, sampler for train
        train_img_paths, train_tag_paths = utils.load_dataset_paths("train")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        trainset = HierarchicalDataset(train_img_paths, train_tag_paths, IMG_CLUSTER_PATH, TAG_CLUSTER_PATH, tokenizer)
        sampler = HierarchicalBatchSampler(batch_size=BATCH_SIZE, dataset=trainset)
        trainloader = torch.utils.data.DataLoader(trainset, sampler=sampler, shuffle=False, num_workers=os.cpu_count(), batch_size=1, pin_memory=True)
        # to calcurate ARR
        train_evalset = utils.EvalDataset(train_img_paths, train_tag_paths, tokenizer)
        train_evalloader = torch.utils.data.DataLoader(train_evalset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
        # for validation
        val_img_paths, val_tag_paths = utils.load_dataset_paths("val")
        valset = utils.EvalDataset(val_img_paths, val_tag_paths, tokenizer)
        valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

        # early stopping
        earlystopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, delta=0)

        # train and save results
        meanARR_best = np.Inf
        for epoch in range(1, MAX_EPOCH + 1):
            print('Epoch {}/{}'.format(epoch, MAX_EPOCH))
            print('-'*10)

            # training and validation
            sampler.set_epoch(epoch)
            loss = train(trainloader, models, temperature, criterions, WEIGHTS, optimizer)
            ARR_train, _ = val(train_evalloader, models, temperature, criterion_CE)
            ARR_val, loss_pair_val = val(valloader, models, temperature, criterion_CE)

            print(f"[train] loss_total: {loss['total']:.4f}, loss_pair: {loss['pair']:.4f}, loss_img: {loss['img']:.4f}, loss_tag: {loss['tag']:.4f}")
            print(f"[train] meanARR: {ARR_train['mean']:.2f}, ARR_tag2img: {ARR_train['tag2img']:.2f}, ARR_img2tag: {ARR_train['img2tag']:.2f}")
            print(f"[val]   meanARR: {ARR_val['mean']:.2f}, ARR_tag2img: {ARR_val['tag2img']:.2f}, ARR_img2tag: {ARR_val['img2tag']:.2f}")
            print(f"[val]   loss_pair: {loss_pair_val:.4f}")
            
            # save results to wandb
            wandb.log({
                    "epoch":             epoch,
                    "loss_total":        loss['total'],
                    "loss_pair":         loss['pair'],
                    "loss_img_epoch":    loss['img'],
                    "loss_tag_epoch":    loss['tag'], 
                    "temperture":        temperature.t.item(),
                    "meanARR_train":     ARR_train['mean'],
                    "ARR_tag2img_train": ARR_train['tag2img'],
                    "ARR_img2tag_train": ARR_train['img2tag'],
                    "meanARR_val":       ARR_val['mean'],
                    "ARR_tag2img_val":   ARR_val['tag2img'],
                    "ARR_img2tag_val":   ARR_val['img2tag'],
                    "val_loss_pair":     loss_pair_val
                    },
                    step = epoch
                    )

            # save results to csv file
            with open(f"{SAVE_DIR}/result_epoch.csv", 'a') as f_epoch:
                csv_writer = csv.writer(f_epoch)
                csv_writer.writerow([epoch,
                                    loss['total'], loss['pair'], loss['img'], loss['tag'], temperature.t.item(),
                                    ARR_train['mean'], ARR_train['tag2img'], ARR_train['img2tag'], 
                                    ARR_val['mean'],   ARR_val['tag2img'],   ARR_val['img2tag'], loss_pair_val])
            
            # early stopping
            earlystopping(ARR_val['mean'])
            if earlystopping.early_stop:
                print("Early stopping")
                filename = f'{SAVE_DIR}/model/checkpoint_{epoch:04d}.pth.tar'
                save_contents = {'font_autoencoder':font_autoencoder.state_dict(), 'clip_model':clip_model.state_dict(),
                                 'emb_i':emb_i.state_dict(), 'emb_t':emb_t.state_dict(), 'optimizer':optimizer.state_dict()}
                torch.save(save_contents, filename)
                break

            # save models and optimizer every 500 epochs
            if epoch%500==0 and epoch!=0:
                filename = f'{SAVE_DIR}/model/checkpoint_{epoch:04d}.pth.tar'
                save_contents = {'font_autoencoder':font_autoencoder.state_dict(), 'clip_model':clip_model.state_dict(),
                                 'emb_i':emb_i.state_dict(), 'emb_t':emb_t.state_dict(), 'optimizer':optimizer.state_dict()}
                torch.save(save_contents, filename)
            
            # save models and optimizer if renew the best meanARR
            if ARR_val['mean']<meanARR_best:
                filename = f"{SAVE_DIR}/model/best.pth.tar"
                save_contents = {'font_autoencoder':font_autoencoder.state_dict(), 'clip_model':clip_model.state_dict(),
                                'emb_i':emb_i.state_dict(), 'emb_t':emb_t.state_dict(), 'optimizer':optimizer.state_dict()}
                torch.save(save_contents, filename)
                meanARR_best = ARR_val['mean']
        
        wandb.log({"meanARR_val_min":meanARR_best}, step=epoch)
        
        wandb.alert(
        title="WandBからの通知", 
        text=f"R={LR}_BS={BATCH_SIZE}_W={WEIGHTS}が終了しました．"
        )


if __name__ == '__main__':

    exp = utils.EXP
    max_epoch = utils.MAX_EPOCH
    early_stopping_patience = utils.EARLY_STOPPING_PATIENCE

    sweep_configuration = {
    'method': 'grid',
    'name': exp.replace('/', '_'),
    'metric': {
        'goal': 'minimize',
        'name': 'meanARR_val_min',
    },
    'parameters': {
        'expt': {
            'value': exp
        },
        'lr': {
            'values': [1e-3, 1e-4]
        },
        'batch_size': {
            'values': [4096, 8192]
        },
        'initial_temperature':{
            'values': [0.07]
        },
        'weights': {    # WEIGHT_PAIR, WEIGHT_IMG, WEIGHT_TAG
            'values': ["[1.0, 1.0, 1.0]"] 
        },
        'max_epoch': {
            'value': max_epoch
        },
        'early_stopping_patience': {
            'value': early_stopping_patience
        },
    }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='Hierarchical_ImpressionCLIP_5-1')
    wandb.agent(sweep_id, main)