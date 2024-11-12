import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
import csv

from HierarchicalDataset import HierarchicalDataset, HierarchicalBatchSampler, EvalDataset
from HierarchicalClipLoss import calc_hierarchical_clip_loss, calc_loss_pair
import ExpMultiplier
import MLP

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


def train(dataloader, models, temperature, criterions, weights, loss_type, ce_bce, optimizer, epoch):
    # one epoch training
    emb_i, emb_t = models
    emb_i.train()
    emb_t.train()

    # Iterate over data
    loss_total_list = []
    loss_pair_list = []
    loss_img_list = []
    loss_tag_list = []
    for idx, data in enumerate(dataloader):
        img_features, tag_features, img_labels, tag_labels = data
        img_features, tag_features, img_labels, tag_labels = img_features[0], tag_features[0], img_labels[0], tag_labels[0]
        img_features = img_features.cuda(non_blocking=True)
        tag_features = tag_features.cuda(non_blocking=True)
        img_labels = img_labels[:,0].cuda(non_blocking=True)
        tag_labels = tag_labels[:,0].cuda(non_blocking=True)
        # prepare labels
        pair_labels = torch.arange(img_features.shape[0]).to('cuda')
        img_labels_transformed = (img_labels.unsqueeze(0)==img_labels.unsqueeze(1)).float()
        tag_labels_transformed = (tag_labels.unsqueeze(0)==tag_labels.unsqueeze(1)).float()
        labels = [pair_labels, img_labels_transformed, tag_labels_transformed]

        # forward
        with torch.set_grad_enabled(True):
            # get model outputs
            embedded_img_features = emb_i(img_features)
            embedded_tag_features = emb_t(tag_features)
            losses = calc_hierarchical_clip_loss(embedded_img_features, embedded_tag_features,
                                                 temperature, weights, criterions, labels, 
                                                 loss_type, ce_bce, epoch)
            loss_total, loss_pair, loss_img, loss_tag = losses
            # append loss to list
            loss_total_list.append(loss_total.item())
            loss_pair_list.append(loss_pair.item())
            loss_img_list.append(loss_img.item())
            loss_tag_list.append(loss_tag.item())
            # backward and optimize
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
    
    # average losses
    aveloss_total = np.mean(loss_total_list)
    aveloss_pair = np.mean(loss_pair_list)
    aveloss_img = np.mean(loss_img_list)
    aveloss_tag = np.mean(loss_tag_list)
    loss_dict = {'total':aveloss_total, 'pair':aveloss_pair, 'img':aveloss_img, 'tag':aveloss_tag}
    
    return loss_dict


def val(dataloader, models, temperature, criterion_CE):
    # one epoch validation
    emb_i, emb_t = models
    emb_i.eval()
    emb_t.eval()

    with torch.no_grad():
        # extruct embedded features
        for idx, data in enumerate(dataloader):
            img_features, tag_features = data
            img_features = img_features.cuda(non_blocking=True)
            tag_features = tag_features.cuda(non_blocking=True)
            # forward
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


if __name__ == '__main__':
    # parameter from config
    params = utils.get_parameters()
    EXPT = params.expt
    MAX_EPOCH = params.max_epoch
    EARLY_STOPPING_PATIENCE = params.early_stopping_patience
    LR = params.learning_rate
    # BATCH_SIZE = params.batch_size.
    BATCH_SIZE = 5
    TEMPERATURE = ['ExpMultiplier', 'ExpMultiplierLogit'][1]
    INITIAL_TEMPERATURE = 0.07
    WEIGHTS = params.weights
    TAG_PREPROCESS = ['normal', 'average_single_tag', 'average_upto_10', 'single_tag'][0]
    LOSS_TYPE = ['average', 'iterative', 'label_and'][2]
    CE_BCE = ['CE', 'BCE'][1]

    BASE_DIR = params.base_dir
    NUM_IMG_CLUSTERS = params.num_img_clusters
    NUM_TAG_CLUSTERS = params.num_tag_clusters
    IMG_CLUSTER_PATH = f'{EXPT}/clustering/cluster/img/train/{NUM_IMG_CLUSTERS}.npz'
    TAG_CLUSTER_PATH = f'{EXPT}/clustering/cluster/tag/{TAG_PREPROCESS}/train/{NUM_TAG_CLUSTERS}.npz'


    # fix random numbers, set cudnn option
    utils.fix_seed(7)
    cudnn.enabled = True
    cudnn.benchmark = True

    # set model and optimized parameters
    device = torch.device('cuda:0')
    emb_i = MLP.ReLU().to(device)
    emb_t = MLP.ReLU().to(device)
    models = [emb_i, emb_t]
    if TEMPERATURE=='ExpMultiplier':
        temperature = ExpMultiplier.ExpMultiplier(INITIAL_TEMPERATURE).to(device)
    elif TEMPERATURE=='ExpMultiplierLogit':
        temperature = ExpMultiplier.ExpMultiplierLogit(INITIAL_TEMPERATURE).to(device)
        
    # set optimizer, criterion
    optimizer = torch.optim.Adam(list(emb_i.parameters())+list(emb_t.parameters())+list(temperature.parameters()), lr=LR)
    criterion_CE = nn.CrossEntropyLoss().to(device)
    criterion_BCE = nn.BCEWithLogitsLoss().to(device)
    criterions = [criterion_CE, criterion_BCE]

    # set dataloder, sampler for train
    img_feature_path_train = f'{EXPT}/feature/img_feature/train.pth'
    tag_feature_path_train = f'{EXPT}/feature/tag_feature/{TAG_PREPROCESS}/train.pth'
    img_feature_path_val = f'{EXPT}/feature/img_feature/val.pth'
    tag_feature_path_val = f'{EXPT}/feature/tag_feature/{TAG_PREPROCESS}/val.pth'
    trainset = HierarchicalDataset(img_feature_path_train, tag_feature_path_train, IMG_CLUSTER_PATH, TAG_CLUSTER_PATH)
    sampler = HierarchicalBatchSampler(batch_size=BATCH_SIZE, dataset=trainset)
    trainloader = torch.utils.data.DataLoader(trainset, sampler=sampler, shuffle=False, num_workers=os.cpu_count(), batch_size=1, pin_memory=True)
    # to calcurate ARR
    train_evalset = EvalDataset(img_feature_path_train, tag_feature_path_train)
    train_evalloader = torch.utils.data.DataLoader(train_evalset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
    # for validation
    valset = EvalDataset(img_feature_path_val, tag_feature_path_val)
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
        loss = train(trainloader, models, temperature, criterions, WEIGHTS, LOSS_TYPE, CE_BCE, optimizer, epoch)
        # ARR_train, _ = val(train_evalloader, models, temperature, criterion_CE)
        # ARR_val, loss_pair_val = val(valloader, models, temperature, criterion_CE)

        # print(f"[train] loss_total: {loss['total']:.4f}, loss_pair: {loss['pair']:.4f}, loss_img: {loss['img']:.4f}, loss_tag: {loss['tag']:.4f}")
        # print(f"[train] meanARR: {ARR_train['mean']:.2f}, ARR_tag2img: {ARR_train['tag2img']:.2f}, ARR_img2tag: {ARR_train['img2tag']:.2f}")
        # print(f"[val]   meanARR: {ARR_val['mean']:.2f}, ARR_tag2img: {ARR_val['tag2img']:.2f}, ARR_img2tag: {ARR_val['img2tag']:.2f}")
        # print(f"[val]   loss_pair: {loss_pair_val:.4f}")

        # # early stopping
        # earlystopping(ARR_val['mean'])
        # if earlystopping.early_stop:
        #     print("Early stopping")
        #     break