"""
pretrain.pyで学習したモデルが出力する特徴量の分布を可視化する
"""

import os
import json
import numpy as np
import torch
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from PIL import Image
import resnet_modified
import seaborn as sns
import matplotlib.pyplot as plt
from openTSNE import TSNE
from sklearn.decomposition import PCA


def set_model(model_path):
    model = resnet_modified.MyResNet(name='resnet50')

    state_dict = torch.load(model_path, map_location='cpu')["state_dict"]
    model_dict = model.state_dict()
    
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k[len("module."):]
        new_state_dict[k] = v
    new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}   # delete unnecessary keys

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    return model


def txt_parse(f):
    result = []
    with open(f) as fp:
        line = fp.readline()
        result.append(line)
        while line:
            line = fp.readline()
            result.append(line)
    return result


class DeepFashionHierarchihcalDataset(Dataset):
    def __init__(self, list_file, class_map_file, repeating_product_file, transform):
        self.transform = transform
        
        with open(list_file, 'r') as f:
            data_dict = json.load(f)

        with open(class_map_file, 'r') as f:
            self.class_map = json.load(f)
        self.repeating_product_ids = txt_parse(repeating_product_file)

        self.filenames = []
        self.category = []
        self.labels = {}
        for i in range(len(data_dict['images'])):
            filename = data_dict['images'][i]
            category = self.class_map[data_dict['categories'][i]]
            product, variation, image = self.get_label_split(filename)
            if product not in self.repeating_product_ids:
                if category not in self.labels:
                    self.labels[category] = {}
                if product not in self.labels[category]:
                    self.labels[category][product] = {}
                if variation not in self.labels[category][product]:
                    self.labels[category][product][variation] = {}
                self.labels[category][product][variation][image] = i
                self.category.append(category)
                self.filenames.append(filename)

    def get_label_split(self, filename):
        split = filename.split('/')
        image_split = split[-1].split('.')[0].split('_')
        return int(split[-2][3:]), int(image_split[0]), int(image_split[1])

    def get_label_split_by_index(self, index):
        filename = self.filenames[index]
        category = self.category[index]
        product, variation, image = self.get_label_split(filename)
        return [category, product, variation, image]

    def __getitem__(self, index):
        image = Image.open(self.filenames[index])
        image = self.transform(image)
        label = list(self.get_label_split_by_index(index))
        return image, torch.tensor(label)

    def __len__(self):
        return len(self.filenames)


# ハイパラ
MODEL_PATH = "results_org/model/checkpoint_0001.pth.tar"
# モデルの準備
device = torch.device('cuda:0')
model2 = set_model(MODEL_PATH).to(device)
pass

# ハイパラ
MODEL_PATH = "results_org/model/checkpoint_0100.pth.tar"
# モデルの準備
device = torch.device('cuda:0')
model1 = set_model(MODEL_PATH).to(device)