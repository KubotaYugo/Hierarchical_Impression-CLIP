import torch
from torch.utils.data.dataset import Dataset
import numpy as np

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


class DatasetForTrain(Dataset):
    def __init__(self, dataset, expt, tag_preprocess):
        # 画像, 印象それぞれの特徴とクラスタのパス
        img_feature_path = f'{expt}/feature/img_feature/{dataset}.pth'
        tag_feature_path = f'{expt}/feature/tag_feature/{tag_preprocess}/{dataset}.pth'
        img_cluster_path = f'{expt}/clustering/clusterID/img/{dataset}.npz'
        tag_cluster_path = f'{expt}/clustering/clusterID/tag/{tag_preprocess}/{dataset}.npz'

        # データの読み込み
        self.img_feature = torch.load(img_feature_path, map_location='cpu')
        self.tag_feature = torch.load(tag_feature_path, map_location='cpu')
        self.img_clusterID = utils.load_hierarchical_clusterID(img_cluster_path)
        self.tag_clusterID = utils.load_hierarchical_clusterID(tag_cluster_path)

    def __len__(self):
        return len(self.img_feature)
    
    def __getitem__(self, idx):
        return self.img_feature[idx], self.tag_feature[idx], self.img_clusterID[idx], self.tag_clusterID[idx]

    
class DatasetForEval(Dataset):
    def __init__(self, dataset, expt, tag_preprocess):
        # 画像, 印象それぞれの特徴とクラスタのパス
        img_feature_path = f'{expt}/feature/img_feature/{dataset}.pth'
        tag_feature_path = f'{expt}/feature/tag_feature/{tag_preprocess}/{dataset}.pth'

        # データの読み込み
        self.img_feature = torch.load(img_feature_path, map_location='cpu')
        self.tag_feature = torch.load(tag_feature_path, map_location='cpu')

    def __len__(self):
        return len(self.img_feature)

    def __getitem__(self, idx):
        return self.img_feature[idx], self.tag_feature[idx]


class ImageDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx])['arr_0'].astype(np.float32)
        img = torch.from_numpy(img/255)
        return img

class ImpressionDataset(Dataset):
    def __init__(self, tag_paths, tag_preprocess):
        self.tag_paths = tag_paths
        self.tag_preprocess = tag_preprocess
    def __len__(self):
        return len(self.tag_paths)
    def __getitem__(self, idx):
        tags = utils.get_font_tags(self.tag_paths[idx])
        prompt = get_prompt(tags)
        return prompt

class SingleTagDataset(Dataset):
    def __init__(self, single_tag_feature_path):
        self.single_tag_feature = torch.load(single_tag_feature_path, map_location='cpu')
    def __len__(self):
        return len(self.single_tag_feature)
    def __getitem__(self, idx):
        return self.single_tag_feature[idx]
    

def get_prompt(tags):
    if len(tags)==1:
        prompt = f'The impression is {tags[0]}.'
    elif len(tags) == 2:
        prompt = f'First and second impressions are {tags[0]} and {tags[1]}, respectively.'
    elif len(tags) >= 3:
        ordinal = ['First', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth']
        prompt1 = ordinal[0]
        prompt2 = tags[0]
        i = 0
        for i in range(1, min(len(tags)-1, 10-1)):
            prompt1 = prompt1 + ', ' + ordinal[i]
            prompt2 = prompt2 + ', ' + tags[i]
        prompt1 = prompt1 + ', and ' + ordinal[i+1] + ' impressions are '
        prompt2 = prompt2 + ', and ' + tags[i+1] + ', respectively.'                
        prompt = prompt1 + prompt2
    return prompt