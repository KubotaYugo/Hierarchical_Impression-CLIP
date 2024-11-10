import torch
from torch.utils.data.dataset import Dataset

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils
from models import HierarchicalDataset


def get_img_features(dataloader, model):
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            input_img = data[0].to('cuda')
            img_feature = model.encoder(input_img)
            if i==0:
                stacked_img_features = img_feature
            else:
                stacked_img_features = torch.concatenate((stacked_img_features, img_feature), dim=0)
        stacked_img_features = stacked_img_features.to("cpu").detach().numpy()
    return stacked_img_features

def get_tag_features(dataloader, model, tokenizer):
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            tokenized_text = tokenizer(data, return_tensors="pt", 
                                       max_length=tokenizer.max_model_input_sizes['openai/clip-vit-base-patch32'], 
                                       padding="max_length", truncation=True)
            inputs = {key: value.to('cuda') for key, value in tokenized_text.items()}
            tag_feature = model.get_text_features(**inputs)
            if i==0:
                stacked_tag_features = tag_feature
            else:
                stacked_tag_features = torch.concatenate((stacked_tag_features, tag_feature), dim=0)
    stacked_tag_features = stacked_tag_features.to('cpu').detach().numpy()
    return stacked_tag_features

def replace_label(label):
    '''
    [2, 2, 1, 4, 5, 3, 2, 5, 4] -> [0, 0, 1, 2, 3, 4, 0, 3, 2]
    のように，リスト先頭から出てくる順に番号を振り直す
    '''
    unique_numbers = {}
    replaced_label = []
    current_number = 0
    for num in label:
        if num not in unique_numbers:
            unique_numbers[num] = current_number
            current_number += 1
        replaced_label.append(unique_numbers[num])
    return replaced_label