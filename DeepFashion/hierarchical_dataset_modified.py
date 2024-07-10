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
import random


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
    def __init__(self, list_file, class_map_file, repeating_product_ids_file, transform=None):
        with open(list_file, 'r') as f:
            data_dict = json.load(f)
        assert len(data_dict['images']) == len(data_dict['categories'])
        num_data = len(data_dict['images'])
        self.transform = transform

        with open(class_map_file, 'r') as f:
            self.class_map = json.load(f)
        self.repeating_product_ids = txt_parse(repeating_product_ids_file)
        self.filenames = []
        self.category = []
        self.labels = {}
        for i in range(num_data):
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
        return category, product, variation, image

    def __getitem__(self, index):
        images0, images1, labels = [], [], []
        for i in index:
            image = Image.open(self.filenames[i])
            label = list(self.get_label_split_by_index(i))
            if self.transform:
                image0, image1 = self.transform(image)
            images0.append(image0)
            images1.append(image1)
            labels.append(label)
        return [torch.stack(images0), torch.stack(images1)], torch.tensor(labels)

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

    def __len__(self):
        return len(self.filenames)


class HierarchicalBatchSampler(Sampler):
    def __init__(self, batch_size: int, drop_last: bool, 
                 dataset: DeepFashionHierarchihcalDataset,) -> None:
        super().__init__(dataset)
        self.batch_size = batch_size
        self.dataset = dataset
        self.epoch=0
        self.drop_last = drop_last

    def random_unvisited_sample(self, label, label_dict, visited, indices, remaining, num_attempt=10):
        attempt = 0
        while attempt < num_attempt:
            idx = self.dataset.random_sample(
                label, label_dict)
            if idx not in visited and idx in indices:
                visited.add(idx)
                return idx
            attempt += 1
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
        
        while len(remaining) > self.batch_size:
            idx = indices[torch.randint(len(indices), (1,))]
            batch.append(idx)
            visited.add(idx)
            category, product, variation, image = self.dataset.get_label_split_by_index(idx)
            
            # メモ: 各階層でidxと異なるデータを被りなしで持ってくる
            image_index = self.random_unvisited_sample(
                image, self.dataset.labels[category][product][variation], visited, indices, remaining)
            variation_index = self.random_unvisited_sample(
                variation, self.dataset.labels[category][product], visited, indices, remaining)
            product_index = self.random_unvisited_sample(
                product, self.dataset.labels[category], visited, indices,  remaining)
            category_index = self.random_unvisited_sample(
                category, self.dataset.labels, visited, indices, remaining)
            
            batch.extend([category_index, product_index, variation_index, image_index])
            visited.update([category_index, product_index, variation_index, image_index])
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


# class DeepFashionHierarchihcalDatasetEval(Dataset):
#     def __init__(self, list_file, class_map_file, repeating_product_ids_file, transform=None):
#         with open(list_file, 'r') as f:
#             data_dict = json.load(f)
#         assert len(data_dict['images']) == len(data_dict['categories'])
#         num_data = len(data_dict['images'])

#         self.transform = transform
#         # self.augment_transform = transforms.RandomChoice([
#         #     transforms.RandomResizedCrop(size=(256, 256), scale=(0.7, 1.)),
#         #     transforms.RandomHorizontalFlip(1),
#         #     transforms.ColorJitter(0.4, 0.4, 0.4)])

#         with open(class_map_file, 'r') as f:
#             self.class_map = json.load(f)
#         self.repeating_product_ids = txt_parse(repeating_product_ids_file)
#         self.filenames = []
#         self.category = []
#         self.labels = {}
#         for i in range(num_data):
#             filename = data_dict['images'][i]

#             category = self.class_map[data_dict['categories'][i]]

#             product, variation, image = self.get_label_split(filename)
#             if product not in self.repeating_product_ids:
#                 if category not in self.labels:
#                     self.labels[category] = {}
#                 if product not in self.labels[category]:
#                     self.labels[category][product] = {}
#                 if variation not in self.labels[category][product]:
#                     self.labels[category][product][variation] = {}
#                 self.labels[category][product][variation][image] = i
#                 self.category.append(category)
#                 self.filenames.append(filename)

#     def get_label_split(self, filename):
#         split = filename.split('/')
#         image_split = split[-1].split('.')[0].split('_')
#         return int(split[-2][3:]), int(image_split[0]), int(image_split[1])

#     def get_label_split_by_index(self, index):
#         filename = self.filenames[index]
#         category = self.category[index]
#         product, variation, image = self.get_label_split(filename)

#         return category, product, variation, image

#     def __getitem__(self, index):
#         image = Image.open(self.filenames[index])
#         label = list(self.get_label_split_by_index(index))
#         if self.transform:
#             image = self.transform(image)

#         return image, label

#     def random_sample(self, label, label_dict):
#         curr_dict = label_dict
#         top_level = True
#         #all sub trees end with an int index
#         while type(curr_dict) is not int:
#             if top_level:
#                 random_label = label
#                 if len(curr_dict.keys()) != 1:
#                     while (random_label == label):
#                         random_label = random.sample(curr_dict.keys(), 1)[0]
#             else:
#                 random_label = random.sample(curr_dict.keys(), 1)[0]
#             curr_dict = curr_dict[random_label]
#             top_level = False
#         return curr_dict

#     def __len__(self):
#         return len(self.filenames)