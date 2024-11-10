'''
4つの前処理方法で印象特徴を保存
normal: 通常の方法(ただタグを並べてエンコード)
average_single_tag: タグ単体でエンコードして平均
avearage_upto_max: タグを最大個数までパディングして平均
    1. 可能な分だけタグを繰り返す
    2. 残った部分には同じタグを複数回使わずに考えられる全通りの組み合わせを入れる
    3. エンコードして平均
single_tag: タグ単体でエンコード(平均しない)
'''

'''
印象特徴を標準化してbisecting kmeansでクラスタリング
'''
import torch
from transformers import CLIPTokenizer, CLIPModel

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils
from models import HierarchicalDataset

def tokenize(prompt):
    tokenized_text = tokenizer(prompt, return_tensors="pt", 
                                max_length=tokenizer.max_model_input_sizes['openai/clip-vit-base-patch32'], 
                                padding="max_length", truncation=True)
    return tokenized_text

def tags_to_prompts(tags, tag_preprocess):
    if tag_preprocess=='normal':
        prompts = [HierarchicalDataset.get_prompt(tags)]
    elif tag_preprocess=='average_single_tag':
        prompts = [HierarchicalDataset.get_prompt([tag]) for tag in tags]
    elif tag_preprocess=='average_upto_10':
        padded_tags = utils.padding_tags(tags)
        prompts = [HierarchicalDataset.get_prompt(tag) for tag in padded_tags]
    return prompts

params = utils.get_parameters()
EXPT = params['expt']
DATASET = params['dataset']
FONTAUTOENCODER_PATH = params['fontautoencoder_path']
BATCH_SIZE = params['batch_size']
TAG_PREPROCESS = params['tag_preprocess']
TAG_FEATURE_PATH = params['tag_feature_path']

SAVE_DIR = os.path.dirname(TAG_FEATURE_PATH)
os.makedirs(SAVE_DIR, exist_ok=True)

# モデルの準備
device = "cuda"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()

# データの準備
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
with torch.no_grad():
    if TAG_PREPROCESS in ['normal', 'average_single_tag', 'average_upto_10']:
        _, tag_paths = utils.load_dataset_paths(DATASET)
        for i, tag_path in enumerate(tag_paths):
            tags = utils.get_font_tags(tag_path)
            prompt = tags_to_prompts(tags, TAG_PREPROCESS)
            tokenized_text = tokenize(prompt)
            inputs = {key: value.to(device) for key, value in tokenized_text.items()}
            tag_feature = model.get_text_features(**inputs)
            tag_feature_ave = torch.mean(tag_feature, dim=0, keepdim=True)
            if i==0:
                stacked_tag_features = tag_feature_ave
            else:
                stacked_tag_features = torch.concatenate((stacked_tag_features, tag_feature_ave), dim=0)
        torch.save(stacked_tag_features, TAG_FEATURE_PATH)
    elif TAG_PREPROCESS=='single_tag':
        tag_list = utils.get_tag_list()
        prompt = [HierarchicalDataset.get_prompt([tag]) for tag in tag_list]
        tokenized_text = tokenize(prompt)
        inputs = {key: value.to(device) for key, value in tokenized_text.items()}
        tag_features = model.get_text_features(**inputs)
        torch.save(tag_features, TAG_FEATURE_PATH)