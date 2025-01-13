import torch
import numpy as np

from HierarchicalClipLoss import calc_hierarchical_clip_loss, calc_loss_eval, calc_loss_pair

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import eval_utils


def train(dataloader, emb_img, emb_tag, temperature, weights, loss_type, optimizer):
    # one epoch training
    emb_img.train()
    emb_tag.train()
    temperature.train()

    # iterate over data
    loss_total_list = []
    loss_img2tag_list = []
    loss_tag2img_list = []
    for data in dataloader:
        # get features and labels
        img_features, tag_features, img_clusterID, tag_clusterID = data
        img_features = img_features.to('cuda')
        tag_features = tag_features.to('cuda')
        img_clusterID = np.asarray(img_clusterID)
        tag_clusterID = np.asarray(tag_clusterID)

        # forward
        with torch.set_grad_enabled(True):
            # get model outputs
            embedded_img_features = emb_img(img_features)
            embedded_tag_features = emb_tag(tag_features)
            loss_dict, layer_loss_dict_img2tag, layer_loss_dict_tag2img = \
                calc_hierarchical_clip_loss(embedded_img_features, embedded_tag_features, 
                                            temperature, weights, img_clusterID, tag_clusterID, loss_type)
            # backward and optimize
            optimizer.zero_grad()
            loss_dict['total'].backward()
            optimizer.step()

            # append loss to list
            loss_total_list.append(loss_dict['total'].item())
            loss_img2tag_list.append(loss_dict['img2tag'].item())
            loss_tag2img_list.append(loss_dict['tag2img'].item())

    # average losses
    aveloss_dict = {
        'total'   :np.mean(loss_total_list), 
        'img2tag' :np.mean(loss_img2tag_list), 
        'tag2img' :np.mean(loss_tag2img_list)
        }
    
    return aveloss_dict, layer_loss_dict_img2tag, layer_loss_dict_tag2img


def train_pair(dataloader, emb_img, emb_tag, temperature, optimizer):
    # one epoch training
    emb_img.train()
    emb_tag.train()
    temperature.train()

    # iterate over data
    loss_pair_total_list = []
    loss_pair_img2tag_list = []
    loss_pair_tag2img_list = []
    for data in dataloader:
        # get features and labels
        img_features, tag_features, _, _ = data
        img_features = img_features.to('cuda')
        tag_features = tag_features.to('cuda')

        # forward
        with torch.set_grad_enabled(True):
            # get model outputs
            embedded_img_features = emb_img(img_features)
            embedded_tag_features = emb_tag(tag_features)
            loss_pair = calc_loss_pair(embedded_img_features, embedded_tag_features, temperature)
        
            # backward and optimize
            optimizer.zero_grad()
            loss_pair['total'].backward()
            optimizer.step()

            # append loss to list
            loss_pair_total_list.append(loss_pair['total'].item())
            loss_pair_img2tag_list.append(loss_pair['img2tag'].item())
            loss_pair_tag2img_list.append(loss_pair['tag2img'].item())

    # average losses
    aveloss_dict = {
        'total'   :np.mean(loss_pair_total_list), 
        'img2tag' :np.mean(loss_pair_img2tag_list), 
        'tag2img' :np.mean(loss_pair_tag2img_list)
        }
    
    return aveloss_dict


def val(dataloader, emb_img, emb_tag, temperature):
    # one epoch validation
    emb_img.eval()
    emb_tag.eval()
    temperature.eval()

    with torch.no_grad():
        # extruct embedded features
        embedded_img_feature_list = []
        embedded_tag_feature_list = []
        for data in dataloader:
            # get features
            img_features, tag_features = data
            img_features = img_features.to('cuda')
            tag_features = tag_features.to('cuda')
            # forward
            embedded_img_features = emb_img(img_features)
            embedded_tag_features = emb_tag(tag_features)
            # append embedded features to list
            embedded_img_feature_list.append(embedded_img_features)
            embedded_tag_feature_list.append(embedded_tag_features)  
        # stack features, calculate losses
        embedded_img_features_stack = torch.concatenate(embedded_img_feature_list, dim=0)
        embedded_tag_features_stack = torch.concatenate(embedded_tag_feature_list, dim=0)
        loss_without_temperature, loss_with_temperature = \
            calc_loss_eval(embedded_img_features, embedded_tag_features, temperature)
        
        # culculate Average Retrieval Rank
        similarity_matrix = torch.matmul(embedded_img_features_stack, embedded_tag_features_stack.T)
        ARR_tag2img = np.mean(eval_utils.retrieval_rank(similarity_matrix, 'tag2img'))
        ARR_img2tag = np.mean(eval_utils.retrieval_rank(similarity_matrix, 'img2tag'))
        meanARR = (ARR_img2tag+ARR_tag2img)/2
        ARR_dict = {
            'mean':   meanARR, 
            'tag2img':ARR_tag2img, 
            'img2tag':ARR_img2tag
            }
        
        embedded_feature_dict = {
            'img': embedded_img_features_stack,
            'tag': embedded_tag_features_stack
        }

    return loss_without_temperature, loss_with_temperature, ARR_dict, embedded_feature_dict


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


def save_models(filename, emb_img, emb_tag, temperature):
    save_contents = {'emb_img':emb_img.state_dict(), 'emb_tag':emb_tag.state_dict(), 'temperature':temperature.state_dict()}
    torch.save(save_contents, filename)

def save_embedded_feature(embedded_feature, save_dir, dataset, epoch):
    torch.save(embedded_feature['img'], f'{save_dir}/embedded_img_feature/{dataset}/{epoch}.pth.tar')
    torch.save(embedded_feature['tag'], f'{save_dir}/embedded_tag_feature/{dataset}/{epoch}.pth.tar')


def loss_format_train(loss):
    text = f'loss_total: {loss['total']:7.4f},  loss_img2tag: {loss['img2tag']:7.4f},  loss_tag2img: {loss['tag2img']:7.4f}'
    return text

def loss_pair_format_train(loss):
    text = f'loss_pair_total: {loss['total']:7.4f},  loss_pair_img2tag: {loss['img2tag']:7.4f},  loss_pair_tag2img: {loss['tag2img']:7.4f}'
    return text

def loss_format_eval(loss):
    text = f'loss_pair: {loss['pair']:7.4f},  loss_pair_img2tag: {loss['pair_img2tag']:7.4f},  loss_pair_tag2img: {loss['pair_tag2img']:7.4f}'
    return text

def ARR_format(ARR):
    text = f'meanARR: {ARR['mean']:7.2f},  ARR_img2tag: {ARR['img2tag']:7.2f},  ARR_tag2img: {ARR['tag2img']:7.2f}'
    return text