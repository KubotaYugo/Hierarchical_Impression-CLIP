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
import copy

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


def get_prompt(tags):
    if len(tags)==1:
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
    return prompt


class HierarchicalDataset(Dataset):
    def __init__(self, img_feature_path, tag_feature_path, img_hierarchy_path, tag_hierarchy_path):
        self.img_feature = torch.load(img_feature_path, map_location='cpu')
        self.tag_feature = torch.load(tag_feature_path, map_location='cpu')
        self.img_hierarchy = np.load(img_hierarchy_path)["arr_0"].astype(np.int64)
        self.tag_hierarchy = np.load(tag_hierarchy_path)["arr_0"].astype(np.int64)
        self.num_data = self.img_feature.shape[0]
        self.img_hierarchical_labels = {}
        self.tag_hierarchical_labels = {}
        for i in range(self.num_data):
            img_cluster = self.img_hierarchy[i]
            tag_cluster = self.tag_hierarchy[i]
            # 画像のモダリティの階層構造
            if img_cluster not in self.img_hierarchical_labels:
                self.img_hierarchical_labels[img_cluster] = {}
            self.img_hierarchical_labels[img_cluster][i] = i       
            # 印象のモダリティの階層構造
            if tag_cluster not in self.tag_hierarchical_labels:
                self.tag_hierarchical_labels[tag_cluster] = {}
            self.tag_hierarchical_labels[tag_cluster][i] = i

    def get_label_split_by_index(self, index, mode):
        if mode=='img':
            cluster = self.img_hierarchy[index]
        elif mode=='tag':
            cluster = self.tag_hierarchy[index]
        font = index
        return cluster, font
    
    def __getitem__(self, idx):
        img_features, tag_features, img_labels, tag_labels = [], [], [], []
        for i in idx:
            # get img feature and tag feature
            img_feature = self.img_feature[i]
            tag_feature = self.tag_feature[i]
            # get labels
            img_label = [self.img_hierarchy[i], i]
            tag_label = [self.tag_hierarchy[i], i]
            # append to list
            img_features.append(img_feature)
            tag_features.append(tag_feature)
            img_labels.append(img_label)
            tag_labels.append(tag_label)
        return torch.stack(img_features), torch.stack(tag_features), torch.tensor(img_labels), torch.tensor(tag_labels)

    def __len__(self):
        return self.num_data


class HierarchicalBatchSampler(Sampler):
    def __init__(self, batch_size: int, dataset: HierarchicalDataset):
        super().__init__(dataset)
        self.batch_size = (batch_size+4) // 5*5
        self.dataset = dataset
        self.length = len(self.dataset) // self.batch_size
        self.epoch=0

    def renew_state(self, idx, batch, visited, remaining, indices):
        if type(idx)==int:
            idx = [idx]
        batch.extend(idx)
        visited.extend(idx)
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

        number_of_iteration = int(self.length*self.batch_size/5)
        for num in range(number_of_iteration):
            idx = remaining[torch.randint(len(remaining), (1,))]
            batch, visited, remaining = self.renew_state(idx, batch, visited, remaining, indices)
            
            # 画像クラスタが同じ/違うフォントをそれぞれピックアップ
            img_cluster, img_font = self.dataset.get_label_split_by_index(idx, 'img')
            img_font_index = self.random_unvisited_sample(img_font, self.dataset.img_hierarchical_labels[img_cluster], visited, indices, remaining)
            batch, visited, remaining = self.renew_state(img_font_index, batch, visited, remaining, indices)
            img_cluster_index = self.random_unvisited_sample(img_cluster, self.dataset.img_hierarchical_labels, visited, indices, remaining)
            batch, visited, remaining = self.renew_state(img_cluster_index, batch, visited, remaining, indices)
            # 印象クラスタが同じ/違うフォントをそれぞれピックアップ
            tag_cluster, tag_font = self.dataset.get_label_split_by_index(idx, 'tag')
            tag_font_index = self.random_unvisited_sample(tag_font, self.dataset.tag_hierarchical_labels[tag_cluster], visited, indices, remaining)
            batch, visited, remaining = self.renew_state(tag_font_index, batch, visited, remaining, indices)
            tag_cluster_index = self.random_unvisited_sample(tag_cluster, self.dataset.tag_hierarchical_labels, visited, indices, remaining)
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
        return self.length


class EvalDataset(Dataset):
    def __init__(self, img_feature_path, tag_feature_path):
        self.img_feature = torch.load(img_feature_path, map_location='cpu')
        self.tag_feature = torch.load(tag_feature_path, map_location='cpu')
    def __len__(self):
        return len(self.img_feature)
    def __getitem__(self, idx):
        return self.img_feature[idx], self.tag_feature[idx]

class SingleTagDataset(Dataset):
    def __init__(self, single_tag_feature_path):
        self.single_tag_feature = torch.load(single_tag_feature_path, map_location='cpu')
    def __len__(self):
        return len(self.single_tag_feature)
    def __getitem__(self, idx):
        return self.single_tag_feature[idx]



class ImageDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx])["arr_0"].astype(np.float32)
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






# class SingleTagDataset(Dataset):
#     def __init__(self, tag_list, tokenizer):
#         self.tag_list = tag_list
#         self.tokenizer = tokenizer
#     def __len__(self):
#         return len(self.tag_list)
#     def __getitem__(self, idx):
#         tag = self.tag_list[idx]
#         prompt = get_prompt([tag])
#         return prompt
#     def __len__(self):
#         return len(self.tag_list)

# class ImpressionDataset(Dataset):
#     def __init__(self, tag_paths, preprocess, tokenizer):
#         self.tag_paths = tag_paths
#         self.tokenizer = tokenizer
#         self.preprocess = preprocess
#     def __len__(self):
#         return len(self.tag_paths)
#     def __getitem__(self, idx):
#         if self.preprocess=='average_single_tag' or self.preprocess=='single_tag':
#             tags = utils.get_font_tags(self.tag_paths[idx])
#             tokenized_tags = [HierarchicalDataset.get_token(self.tokenizer, [tag]) for tag in tags]
#         elif self.preprocess=='average_upto_max':
#             tags = utils.get_font_tags(self.tag_paths[idx])
#             tags_padded = padding_tags(tags)
#             tokenized_tags = HierarchicalDataset.get_token(self.tokenizer, tags_padded)
#         return tokenized_tags


# class ImpressionDataset(Dataset):
#     def __init__(self, tag_paths, tokenizer):
#         self.tag_paths = tag_paths
#         self.tokenizer = tokenizer
#     def __len__(self):
#         return len(self.tag_paths)
#     def __getitem__(self, idx):
#         tags = utils.get_font_tags(self.tag_paths[idx])
#         tokenized_tags = get_token(self.tokenizer, tags)
#         return tokenized_tags

# class HierarchicalDatasetWithoutSampler(Dataset):
#     def __init__(self, img_paths, tag_paths, img_hierarchy_path, tag_hierarchy_path, tokenizer):
#         self.tokenizer = tokenizer
#         self.img_paths = img_paths
#         self.tag_paths = tag_paths
#         self.img_hierarchy = np.load(img_hierarchy_path)["arr_0"].astype(np.int64)
#         self.tag_hierarchy = np.load(tag_hierarchy_path)["arr_0"].astype(np.int64)
#         self.num_data = len(img_paths)
#         self.img_labels = {}
#         self.tag_labels = {}
#         for i in range(self.num_data):
#             img_cluster = self.img_hierarchy[i]
#             tag_cluster = self.tag_hierarchy[i]
#             # 画像のモダリティの階層構造
#             if img_cluster not in self.img_labels:
#                 self.img_labels[img_cluster] = {}
#             self.img_labels[img_cluster][i] = i       
#             # 印象のモダリティの階層構造
#             if tag_cluster not in self.tag_labels:
#                 self.tag_labels[tag_cluster] = {}
#             self.tag_labels[tag_cluster][i] = i
        
#     def __getitem__(self, idx):
#         img = np.load(self.img_paths[idx])["arr_0"].astype(np.float32)
#         img = torch.from_numpy(img/255)
#         tokenized_tag = get_token(self.tokenizer, self.tag_paths[idx])
#         img_label = [self.img_hierarchy[idx], idx]
#         tag_label = [self.tag_hierarchy[idx], idx]
#         return img, torch.tensor(tokenized_tag), torch.tensor(img_label), torch.tensor(tag_label)

#     # def __getitem__(self, idx):
#     #     imgs, tokenized_tags, img_labels, tag_labels = [], [], [], []
#     #     img = np.load(self.img_paths[idx])["arr_0"].astype(np.float32)
#     #     img = torch.from_numpy(img/255)
#     #     tokenized_tag = get_token(self.tag_paths[idx])
#     #     img_label = [self.img_hierarchy[idx], idx]
#     #     tag_label = [self.tag_hierarchy[idx], idx]
#     #     imgs.append(img)
#     #     tokenized_tags.append(tokenized_tag)
#     #     img_labels.append(img_label)
#     #     tag_labels.append(tag_label)
#     #     return imgs, tokenized_tags, torch.tensor(img_labels), torch.tensor(tag_labels)

#     def __len__(self):
#         return self.num_data


# class EvalDataset(Dataset):
#     def __init__(self, img_paths, tag_paths, tokenizer):
#         self.img_paths = img_paths
#         self.tag_paths = tag_paths
#         self.tokenizer = tokenizer
#     def __len__(self):
#         return len(self.img_paths)
#     def __getitem__(self, idx):
#         img = np.load(self.img_paths[idx])["arr_0"].astype(np.float32)
#         img = torch.from_numpy(img/255)
#         tags = utils.get_font_tags(self.tag_paths[idx])
#         tokenized_tags = get_token(self.tokenizer, tags)
#         return img, tokenized_tags


# class EvalDataset(Dataset):
#     def __init__(self, img_paths=None, tag_paths=None, img_hierarchy_path=None, tag_hierarchy_path=None, tag_list=None, tag_preprocess=None):
#         self.img_paths = img_paths
#         self.tag_paths = tag_paths
#         self.img_hierarchy_path = img_hierarchy_path
#         self.tag_hierarchy_path = tag_hierarchy_path
#         self.tag_list = tag_list
#         self.tag_preprocess = tag_preprocess

#         if self.img_hierarchy_path!=None:
#             self.img_hierarchy = np.load(self.img_hierarchy_path)["arr_0"].astype(np.int64)        
#         if self.tag_hierarchy_path!=None:
#             self.tag_hierarchy = np.load(self.tag_hierarchy_path)["arr_0"].astype(np.int64)

#         if self.img_paths!=None:
#             self.num_data = len(self.img_paths)
#         elif self.tag_paths!=None:
#             self.num_data = len(self.tag_paths)
#         elif self.img_hierarchy_path!=None:
#             self.num_data = len(self.img_hierarchy_path)
#         elif self.tag_hierarchy_path!=None:
#             self.num_data = len(self.tag_hierarchy_path)
#         elif self.tag_list!=None:
#             self.num_data = len(self.tag_list)
        
#     def __getitem__(self, idx):
#         return_list = []
#         if self.img_paths!=None:
#             img = np.load(self.img_paths[idx])["arr_0"].astype(np.float32)
#             img = torch.from_numpy(img/255)
#             return_list.append(img)
#         if self.tag_paths!=None:
#             tags = utils.get_font_tags(self.tag_paths[idx])
#             if self.tag_preprocess=='normal':
#                 prompt = get_prompt(tags)
#             elif self.tag_preprocess=='average_upto_10':
#                 padded_tags = padding_tags(tags)
#                 prompt = get_prompt(padded_tags)
#             elif self.tag_preprocess=='average_single_tag':
#                 prompt = [get_prompt([tag]) for tag in tags]
#             return_list.append(prompt)
#         if self.img_hierarchy_path!=None:
#             img_label = [self.img_hierarchy[idx], idx]
#             return_list.append(img_label)
#         if self.tag_hierarchy_path!=None:
#             tag_label = [self.tag_hierarchy[idx], idx]
#             return_list.append(tag_label)
#         if self.tag_list!=None and self.tokenizer!=None:
#             tag = self.tag_list[idx]
#             prompt = f"The impression is {tag}."
#             tokenized_text = self.tokenizer(prompt, return_tensors="pt", max_length=self.tokenizer.max_model_input_sizes['openai/clip-vit-base-patch32'], padding="max_length", truncation=True)['input_ids'][0]     
#             return_list.append(tokenized_text)
#         return return_list
    
#     def __len__(self):
#         return self.num_data