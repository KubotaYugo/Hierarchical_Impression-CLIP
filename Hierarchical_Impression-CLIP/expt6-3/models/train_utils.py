import torch
import numpy as np

from HierarchicalClipLoss import calc_hierarchical_clip_loss, calc_hierarchical_clip_loss_eval

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import eval_utils


def train(dataloader, emb_img, emb_tag, temperature, criterions, weights, loss_type, ce_bce, optimizer, epoch):
    # one epoch training
    emb_img.train()
    emb_tag.train()
    temperature.train()

    # iterate over data
    loss_total_list = []
    loss_pair_list = []
    loss_img_list = []
    loss_tag_list = []
    for idx, data in enumerate(dataloader):
        # get features and labels
        img_features, tag_features, img_labels, tag_labels = data
        img_features, tag_features, img_labels, tag_labels = img_features[0], tag_features[0], img_labels[0], tag_labels[0]
        img_features = img_features.to('cuda')
        tag_features = tag_features.to('cuda')
        img_labels = img_labels[:,0].to('cuda')
        tag_labels = tag_labels[:,0].to('cuda')

        # prepare labels
        pair_labels = torch.arange(img_features.shape[0]).to('cuda')
        img_labels_transformed = (img_labels.unsqueeze(0)==img_labels.unsqueeze(1)).float()
        tag_labels_transformed = (tag_labels.unsqueeze(0)==tag_labels.unsqueeze(1)).float()
        labels = [pair_labels, img_labels_transformed, tag_labels_transformed]

        # forward
        with torch.set_grad_enabled(True):
            # get model outputs
            embedded_img_features = emb_img(img_features)
            embedded_tag_features = emb_tag(tag_features)
            loss_dict = calc_hierarchical_clip_loss(embedded_img_features, embedded_tag_features, 
                                                    temperature, weights, criterions, labels, 
                                                    loss_type, ce_bce, epoch)

            # backward and optimize
            optimizer.zero_grad()
            loss_dict['total'].backward()
            optimizer.step()

            # append loss to list
            loss_total_list.append(loss_dict['total'].item())
            loss_pair_list.append(loss_dict['pair'].item())
            loss_img_list.append(loss_dict['img'].item())
            loss_tag_list.append(loss_dict['tag'].item())

    # average losses
    aveloss_dict = {
        'total':np.mean(loss_total_list), 
        'pair' :np.mean(loss_pair_list), 
        'img'  :np.mean(loss_img_list), 
        'tag'  :np.mean(loss_tag_list)
        }
    
    return aveloss_dict


def val(dataloader, emb_img, emb_tag, temperature, criterions, weights, ce_bce):
    # one epoch validation
    emb_img.eval()
    emb_tag.eval()
    temperature.eval()

    with torch.no_grad():
        # extruct embedded features
        img_labels_list = []
        tag_labels_list = []
        embedded_img_feature_list = []
        embedded_tag_feature_list = []
        for data in dataloader:
            # get features and labels
            img_features, tag_features, img_labels, tag_labels = data
            img_features = img_features.to('cuda')
            tag_features = tag_features.to('cuda')
            img_labels = torch.stack(img_labels, dim=1)[:,0].to('cuda')
            tag_labels = torch.stack(tag_labels, dim=1)[:,0].to('cuda')
            # forward
            embedded_img_features = emb_img(img_features)
            embedded_tag_features = emb_tag(tag_features)
            # append labels and embedded features to list
            img_labels_list.append(img_labels)
            tag_labels_list.append(tag_labels)
            embedded_img_feature_list.append(embedded_img_features)
            embedded_tag_feature_list.append(embedded_tag_features)

        # prepare labels
        img_labels_stack = torch.concatenate(img_labels_list, dim=0)
        tag_labels_stack = torch.concatenate(tag_labels_list, dim=0)
        img_labels_transformed = (img_labels_stack.unsqueeze(0)==img_labels_stack.unsqueeze(1)).float()
        tag_labels_transformed = (tag_labels_stack.unsqueeze(0)==tag_labels_stack.unsqueeze(1)).float()
        pair_labels = torch.arange(img_labels_stack.shape[0]).to('cuda')
        labels = [pair_labels, img_labels_transformed, tag_labels_transformed]     
        # stack features, calcurate losses
        embedded_img_features_stack = torch.concatenate(embedded_img_feature_list, dim=0)
        embedded_tag_features_stack = torch.concatenate(embedded_tag_feature_list, dim=0)
        loss_without_temperature, loss_with_temperature = calc_hierarchical_clip_loss_eval(
                                                                embedded_img_features_stack, embedded_tag_features_stack, 
                                                                temperature, weights, criterions, labels, ce_bce)
        
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
        
    return loss_without_temperature, loss_with_temperature, ARR_dict


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

def loss_format(loss):
    text = f'loss_total: {loss['total']:7.4f},  loss_pair: {loss['pair']:7.4f},  loss_img: {loss['img']:7.4f},  loss_tag: {loss['tag']:7.4f}'
    return text