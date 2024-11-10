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


class Dataset_for_SimCLR(Dataset):
    def __init__(self, img_paths, tag_paths, tokenizer):
        self.tokenizer = tokenizer
        self.img_paths = img_paths
        self.tag_paths = tag_paths
        self.num_data = len(img_paths)

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
        img = np.load(self.img_paths[index])["arr_0"].astype(np.float32)
        img = torch.from_numpy(img/255)
        tokenized_tag = self.get_token(self.tag_paths[index])['input_ids'][0]
        return img, tokenized_tag

    def __len__(self):
        return self.num_data