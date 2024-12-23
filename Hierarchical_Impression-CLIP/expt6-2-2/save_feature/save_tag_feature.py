'''
4つの前処理方法で印象特徴を保存
normal: 通常の方法(ただタグを並べてエンコード)
average_single_tag: タグ単体でエンコードして平均
avearage_upto_max: 以下の方法でタグを最大個数までパディングして平均
    1. 可能な分だけタグを繰り返す
    2. 残った部分には同じタグを複数回使わずに考えられる全通りの組み合わせを入れる
    3. エンコードして平均
single_tag: タグ単体でエンコード(平均しない)
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

def tags_to_prompts(tags, tag_preprocess):
    if tag_preprocess=='normal':
        prompts = [HierarchicalDataset.get_prompt(tags)]
    elif tag_preprocess=='average_single_tag':
        prompts = [HierarchicalDataset.get_prompt([tag]) for tag in tags]
    elif tag_preprocess=='average_upto_10':
        padded_tags = padding_tags(tags)
        prompts = [HierarchicalDataset.get_prompt(tag) for tag in padded_tags]
    return prompts


params = utils.get_parameters()
DATASET = params.dataset
TAG_PREPROCESS = params.tag_preprocess
TAG_FEATURE_PATH = params.tag_feature_path

SAVE_DIR = os.path.dirname(TAG_FEATURE_PATH)
os.makedirs(SAVE_DIR, exist_ok=True)

# モデルの準備
device = torch.device('cuda:0')
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
model.eval()

# 特徴量をエンコードして保存
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
if TAG_PREPROCESS in ['normal', 'average_single_tag', 'average_upto_10']:
    _, tag_paths = utils.load_dataset_paths(DATASET)
    tag_feature_list = []
    for tag_path in tag_paths:
        # タグをプロンプトに変換
        tags = utils.get_font_tags(tag_path)
        prompt = tags_to_prompts(tags, TAG_PREPROCESS)
        tokenized_text = tokenize(prompt)
        # エンコード
        inputs = {key: value.to(device) for key, value in tokenized_text.items()}
        with torch.no_grad():
            tag_feature = model.get_text_features(**inputs)
            tag_feature_ave = torch.mean(tag_feature, dim=0, keepdim=True)
        tag_feature_list.append(tag_feature_ave)
    stacked_tag_feature = torch.cat(tag_feature_list, dim=0)
    torch.save(stacked_tag_feature, TAG_FEATURE_PATH)

elif TAG_PREPROCESS=='single_tag':
    # タグをプロンプトに変換
    tag_list = utils.get_tag_list()
    prompt = [HierarchicalDataset.get_prompt([tag]) for tag in tag_list]
    tokenized_text = tokenize(prompt)
    # エンコード
    inputs = {key: value.to(device) for key, value in tokenized_text.items()}
    with torch.no_grad():
        tag_feature = model.get_text_features(**inputs)
    torch.save(tag_feature, TAG_FEATURE_PATH)