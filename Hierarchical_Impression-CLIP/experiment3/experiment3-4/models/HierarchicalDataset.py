'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import random
import csv
import copy


class HierarchicalDataset(Dataset):
    def __init__(self, img_paths, tag_paths, img_hierarchy_path, tag_hierarchy_path, tokenizer):
        self.tokenizer = tokenizer
        self.img_paths = img_paths
        self.tag_paths = tag_paths
        self.img_hierarchy = np.load(img_hierarchy_path)["arr_0"].astype(np.int64)
        self.tag_hierarchy = np.load(tag_hierarchy_path)["arr_0"].astype(np.int64)
        self.num_data = len(img_paths)
        self.img_labels = {}
        self.tag_labels = {}
        for i in range(self.num_data):
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
        tokenized_text = self.tokenizer(prompt, return_tensors="pt", 
                                        max_length=self.tokenizer.max_model_input_sizes['openai/clip-vit-base-patch32'], 
                                        padding="max_length", truncation=True)
        return tokenized_text
    
    def get_label_split_by_index(self, index, mode):
        if mode=='img':
            cluster = self.img_hierarchy[index]
        elif mode=='tag':
            cluster = self.tag_hierarchy[index]
        font = index
        return cluster, font

    def __getitem__(self, index):
        imgs, tokenized_tags, img_labels, tag_labels = [], [], [], []
        for i in index:
            img = np.load(self.img_paths[i])["arr_0"].astype(np.float32)
            img = torch.from_numpy(img/255)
            tokenized_tag = self.get_token(self.tag_paths[i])['input_ids'][0]
            img_label = [self.img_hierarchy[i], i]  # メモ: [self.img_hierarchy[i], i, i]になっていたのを変更
            tag_label = [self.tag_hierarchy[i], i]  # メモ: [self.tag_hierarchy[i], i, i]になっていたのを変更
            imgs.append(img)
            tokenized_tags.append(tokenized_tag)
            img_labels.append(img_label)
            tag_labels.append(tag_label)
        return torch.stack(imgs), torch.stack(tokenized_tags), torch.tensor(img_labels), torch.tensor(tag_labels)

    def __len__(self):
        return self.num_data


class HierarchicalBatchSampler(Sampler):
    def __init__(self, batch_size: int, drop_last: bool, dataset: HierarchicalDataset):
        super().__init__(dataset)
        self.batch_size = batch_size
        self.dataset = dataset
        self.epoch=0
        self.drop_last = drop_last

    def renew_state(self, index, batch, visited, remaining, indices):
        if type(index)==int:
            index = [index]
        batch.extend(index)
        visited.extend(index)
        remaining = list(set(indices).difference(visited))
        return batch, visited, remaining

    def random_sample(self, label, label_dict):
        '''
        label_dictの中からlabel以外のクラスタのインデックスをランダムに選択
        '''
        curr_dict = label_dict
        top_level = True
        while type(curr_dict) is not int:   # all sub trees end with an int index
            if top_level:
                random_label = label
                if len(curr_dict.keys()) != 1:
                    while (random_label == label):  # メモ: to avoid choosing the category same as chosen one in the target category
                        random_label = random.sample(list(curr_dict.keys()), 1)[0]
            else:
                random_label = random.sample(list(curr_dict.keys()), 1)[0]
            curr_dict = curr_dict[random_label]
            top_level = False
        return curr_dict

    def random_unvisited_sample(self, label, label_dict, visited, indices, remaining, num_attempt=50):
        attempt = 0
        while attempt < num_attempt:
            idx = self.random_sample(label, label_dict)
            if idx not in visited and idx in remaining:   # メモ: indices->remainingに変更した
                return idx
            attempt += 1
        # 上の条件を満たすものを得られなかった場合，残りからランダムに選択
        idx = remaining[torch.randint(len(remaining), (1,))]
        return idx

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        batch = []
        visited = []
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        remaining = copy.deepcopy(indices)

        while len(remaining) > self.batch_size: # まだミニバッチを作れるなら実行
            idx = remaining[torch.randint(len(remaining), (1,))]
            batch, visited, remaining = self.renew_state(idx, batch, visited, remaining, indices)
            
            # 画像クラスタが同じ/違うフォントをそれぞれピックアップ
            img_cluster, img_font = self.dataset.get_label_split_by_index(idx, 'img')
            img_font_index = self.random_unvisited_sample(img_font, self.dataset.img_labels[img_cluster], visited, indices, remaining)
            batch, visited, remaining = self.renew_state(img_font_index, batch, visited, remaining, indices)
            img_cluster_index = self.random_unvisited_sample(img_cluster, self.dataset.img_labels, visited, indices, remaining)
            batch, visited, remaining = self.renew_state(img_cluster_index, batch, visited, remaining, indices)
            # 印象クラスタが同じ/違うフォントをそれぞれピックアップ
            tag_cluster, tag_font = self.dataset.get_label_split_by_index(idx, 'tag')
            tag_font_index = self.random_unvisited_sample(tag_font, self.dataset.tag_labels[tag_cluster], visited, indices, remaining)
            batch, visited, remaining = self.renew_state(tag_font_index, batch, visited, remaining, indices)
            tag_cluster_index = self.random_unvisited_sample(tag_cluster, self.dataset.tag_labels, visited, indices, remaining)
            batch, visited, remaining = self.renew_state(tag_cluster_index, batch, visited, remaining, indices)

            assert len(batch) == len(set(batch)), "Same index in batch!"
            assert len(visited) == len(set(visited)), "Same index in visited!"

            # バッチ数分溜まったらバッチを渡す
            if len(batch) >= self.batch_size:
                yield batch
                batch = []

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size