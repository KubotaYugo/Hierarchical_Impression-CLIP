'''
タグ単体でエンコードしたときのL2ノルムの大きさを見る
'''
import torch
import itertools
from transformers import CLIPTokenizer, CLIPModel

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils
from models import HierarchicalDataset


def tokenize(prompt):
    tokenized_text = tokenizer(prompt, return_tensors='pt', 
                                max_length=tokenizer.max_model_input_sizes['openai/clip-vit-base-patch32'], 
                                padding='max_length', truncation=True)
    return tokenized_text

def padding_tags(tags_org, max_length=10):
    '''
    tags_org = ['a', 'b', 'c'], max_length=5のとき, 以下のようにmax_lengthまでパディングする
    [['a', 'b', 'c', 'a', 'b'], ['a', 'b', 'c', 'a', 'c'], ['a', 'b', 'c', 'b', 'a'], 
     ['a', 'b', 'c', 'b', 'c'], ['a', 'b', 'c', 'c', 'a'], ['a', 'b', 'c', 'c', 'b']]
    '''
    tags = tags_org
    while len(tags)+len(tags_org)<=max_length:
        tags = tags+tags_org
    all_combinations = itertools.product(tags_org, repeat=max_length-len(tags))
    add = [list(p) for p in all_combinations if len(set(p)) == len(p)]
    if add!=['']:
        return_list = [tags+a for a in add]
    else:
        return_list = tags
    return return_list

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

def tags_to_prompts(tags, tag_preprocess):
    if tag_preprocess=='normal':
        prompts = [get_prompt(tags)]
    elif tag_preprocess=='average_single_tag':
        prompts = [get_prompt([tag]) for tag in tags]
    elif tag_preprocess=='average_upto_10':
        padded_tags = padding_tags(tags)
        prompts = [get_prompt(tag) for tag in padded_tags]
    return prompts




params = utils.get_parameters()
DATASET = 'val'
TAG_PREPROCESS = 'single_tag'

# モデルの準備
device = torch.device('cuda:0')
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
model.eval()
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')

# タグをプロンプトに変換
tag_list = utils.get_tag_list()
prompt = [HierarchicalDataset.get_prompt([tag]) for tag in tag_list]
tokenized_text = tokenize(prompt)
# エンコード
inputs = {key: value.to(device) for key, value in tokenized_text.items()}
with torch.no_grad():
    tag_feature = model.get_text_features(**inputs)
    tag_feature = tag_feature / tag_feature.norm(dim=-1, keepdim=True)
    similarity = torch.matmul(tag_feature, tag_feature.t())
    pass