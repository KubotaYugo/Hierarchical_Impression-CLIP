'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import json
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from PIL import Image
import numpy as np
import random
import os
import csv


class DMH_D(Dataset):
    def __init__(self, img_paths, tag_paths, img_hierarchy_path, tag_hierarchy_path, tokenizer):
        self.tokenizer = tokenizer
        self.img_paths = img_paths
        self.tag_paths = tag_paths
        self.img_hierarchy = np.load(img_hierarchy_path)["arr_0"].astype(np.int64)
        self.tag_hierarchy = np.load(tag_hierarchy_path)["arr_0"].astype(np.int64)
        self.num_data = len(img_paths)
        # self.fontnames = [os.path.basename(i)[:-4] for i in img_paths]
        self.img_labels = {}
        self.tag_labels = {}
        for i in range(self.num_data):
            # fontname = self.fontnames[i]
            img_cluster = self.img_hierarchy[i]
            tag_cluster = self.tag_hierarchy[i]
            # 画像のモダリティの階層構造
            if img_cluster not in self.img_labels:
                self.img_labels[img_cluster] = {}
            self.img_labels[img_cluster][i] = i       
            # 印象のモダリティの階層構造
            if tag_cluster not in self.tag_labels:
                self.tag_labels[tag_cluster] = {}
            self.tag_labels[tag_cluster][i] = i

    def get_token(self, tag_path):
        with open(tag_path, encoding='utf8') as f:
            csvreader = csv.reader(f)
            tags = [row for row in csvreader][0]
        if len(tags) == 1:
            prompt = f"The impression is {tags[0]}."
        elif len(tags) == 2:
            prompt = f"First and second impressions are {tags[0]} and {tags[1]}, respectively."
        elif len(tags) >= 3:
            ordinal = ["First", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
            prompt1 = ordinal[0]
            prompt2 = tags[0]
            i = 0
            for i in range(1, min(len(tags)-1, 10-1)):
                prompt1 = prompt1 + ", " + ordinal[i]
                prompt2 = prompt2 + ", " + tags[i]
            prompt1 = prompt1 + ", and " + ordinal[i+1] + " impressions are "
            prompt2 = prompt2 + ", and " + tags[i+1] + ", respectively."                
            prompt = prompt1 + prompt2
        tokenized_text = self.tokenizer(prompt, return_tensors="pt", max_length=self.tokenizer.max_model_input_sizes['openai/clip-vit-base-patch32'], padding="max_length", truncation=True)
        return tokenized_text
        
    def __getitem__(self, index):
        imgs, tokenized_tags, img_labels, tag_labels = [], [], [], []
        for i in index:
            img = np.load(self.img_paths[i])["arr_0"].astype(np.float32)
            img = torch.from_numpy(img/255)
            tokenized_tag = self.get_token(self.tag_paths[i])['input_ids'][0]
            img_label = [self.img_hierarchy[i], i]
            tag_label = [self.tag_hierarchy[i], i]
            imgs.append(img)
            tokenized_tags.append(tokenized_tag)
            img_labels.append(img_label)
            tag_labels.append(tag_label)
        return torch.stack(imgs), torch.stack(tokenized_tags), torch.tensor(img_labels), torch.tensor(tag_labels)

    def __len__(self):
        return self.num_data
    
    
    def random_sample(self, label, label_dict):
        curr_dict = label_dict
        top_level = True
        # all sub trees end with an int index
        while type(curr_dict) is not int:
            if top_level:
                random_label = label
                if len(curr_dict.keys()) != 1:
                    # メモ: to avoid choosing the category same as chosen one in the target category
                    while (random_label == label):
                        random_label = random.sample(list(curr_dict.keys()), 1)[0]
            else:
                random_label = random.sample(list(curr_dict.keys()), 1)[0]
            curr_dict = curr_dict[random_label]
            top_level = False
        return curr_dict


class DMH_BS(Sampler):
    def __init__(self, batch_size: int, drop_last: bool, dataset: DMH_D):
        super().__init__(dataset)
        self.batch_size = batch_size
        self.dataset = dataset
        self.epoch=0
        self.drop_last = drop_last
    
    def random_unvisited_sample(self, label, label_dict, visited, indices, remaining, num_attempt=50):
        attempt = 0
        while attempt < num_attempt:
            idx = self.dataset.random_sample(label, label_dict)
            if idx not in visited and idx in indices:
                visited.add(idx)
                return idx
            attempt += 1
        # 上の条件を満たすものを得られなかった場合，残りからランダムに選択
        idx = remaining[torch.randint(len(remaining), (1,))]
        visited.add(idx)
        return idx

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        batch = []
        visited = set()
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        remaining = list(set(indices).difference(visited))
        idx_list = []

        while len(remaining) > self.batch_size:
            # メモ: idxはvisitedにある値になることもある(1epoch中に同じデータが使われる)
            # ここをremainingからサンプリングするようにすれば，同じデータが使われることがなくなる？
            # idxとして同じ数字が複数回選ばれることもある(修正した方がよさそう)
            idx = indices[torch.randint(len(indices), (1,))]
            idx_list.append(idx)
            batch.append(idx)
            visited.add(idx)
            img_cluster = self.dataset.img_hierarchy[idx]
            tag_cluster = self.dataset.tag_hierarchy[idx]
            
            # メモ: 各階層でidxと異なるデータを被りなしで持ってくる
            # 画像の階層構造と印象の階層構造で同じものを持ってこないようにする
            img_cluster_index = self.random_unvisited_sample(
                idx, self.dataset.img_labels[img_cluster], visited, indices, remaining)
            tag_cluster_index = self.random_unvisited_sample(
                idx, self.dataset.tag_labels[tag_cluster], visited, indices, remaining)
    
            batch.extend([img_cluster_index, tag_cluster_index])
            visited.update([img_cluster_index, tag_cluster_index])
            remaining = list(set(indices).difference(visited))

            if len(batch) >= self.batch_size:
                yield batch
                batch = []
            remaining = list(set(indices).difference(visited))

        if (len(remaining) > self.batch_size) and not self.drop_last:
            batch.update(list(remaining))
            yield batch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size
    

class DMH_D_Eval(Dataset):
    def __init__(self, img_paths, tag_paths, img_hierarchy_path, tag_hierarchy_path, tokenizer):
        self.tokenizer = tokenizer
        self.img_paths = img_paths
        self.tag_paths = tag_paths
        self.img_hierarchy = np.load(img_hierarchy_path)["arr_0"].astype(np.int64)
        self.tag_hierarchy = np.load(tag_hierarchy_path)["arr_0"].astype(np.int64)
        self.num_data = len(img_paths)
        # self.fontnames = [os.path.basename(i)[:-4] for i in img_paths]
        self.img_labels = {}
        self.tag_labels = {}
        for i in range(self.num_data):
            # fontname = self.fontnames[i]
            img_cluster = self.img_hierarchy[i]
            tag_cluster = self.tag_hierarchy[i]
            # 画像のモダリティの階層構造
            if img_cluster not in self.img_labels:
                self.img_labels[img_cluster] = {}
            self.img_labels[img_cluster][i] = i       
            # 印象のモダリティの階層構造
            if tag_cluster not in self.tag_labels:
                self.tag_labels[tag_cluster] = {}
            self.tag_labels[tag_cluster][i] = i

    def get_token(self, tag_path):
        with open(tag_path, encoding='utf8') as f:
            csvreader = csv.reader(f)
            tags = [row for row in csvreader][0]
        if len(tags) == 1:
            prompt = f"The impression is {tags[0]}."
        elif len(tags) == 2:
            prompt = f"First and second impressions are {tags[0]} and {tags[1]}, respectively."
        elif len(tags) >= 3:
            ordinal = ["First", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
            prompt1 = ordinal[0]
            prompt2 = tags[0]
            i = 0
            for i in range(1, min(len(tags)-1, 10-1)):
                prompt1 = prompt1 + ", " + ordinal[i]
                prompt2 = prompt2 + ", " + tags[i]
            prompt1 = prompt1 + ", and " + ordinal[i+1] + " impressions are "
            prompt2 = prompt2 + ", and " + tags[i+1] + ", respectively."                
            prompt = prompt1 + prompt2
        tokenized_text = self.tokenizer(prompt, return_tensors="pt", max_length=self.tokenizer.max_model_input_sizes['openai/clip-vit-base-patch32'], padding="max_length", truncation=True)
        return tokenized_text
        
    def __getitem__(self, index):
        imgs, tokenized_tags, img_labels, tag_labels = [], [], [], []
        img = np.load(self.img_paths[index])["arr_0"].astype(np.float32)
        img = torch.from_numpy(img/255)
        tokenized_tag = self.get_token(self.tag_paths[index])['input_ids'][0]
        img_label = [self.img_hierarchy[index], index]
        tag_label = [self.tag_hierarchy[index], index]
        imgs.append(img)
        tokenized_tags.append(tokenized_tag)
        img_labels.append(img_label)
        tag_labels.append(tag_label)
        return imgs, tokenized_tags, torch.tensor(img_labels), torch.tensor(tag_labels)

    def __len__(self):
        return self.num_data
