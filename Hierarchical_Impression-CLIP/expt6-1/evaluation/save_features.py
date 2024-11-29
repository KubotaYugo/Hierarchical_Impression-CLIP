'''
共埋め込み後の画像特徴と印象特徴を保存
'''
import torch

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from models import MLP
from models import HierarchicalDataset
from lib import utils


def extract_feature(dataloader, emb_i, emb_t):
    img_feature_list = []
    tag_feature_list = []
    embedded_tag_feature_list = []
    embedded_img_feature_list = []
    for data in dataloader:
        img_feature, tag_feature = data
        img_feature = img_feature.cuda(non_blocking=True)
        tag_feature = tag_feature.cuda(non_blocking=True)
        with torch.no_grad():
            embedded_img_feature = emb_i(img_feature)
            embedded_tag_feature = emb_t(tag_feature)
        img_feature_list.append(img_feature)
        tag_feature_list.append(tag_feature)
        embedded_img_feature_list.append(embedded_img_feature)
        embedded_tag_feature_list.append(embedded_tag_feature)
    img_feature_stack = torch.concatenate(img_feature_list, dim=0)
    tag_feature_stack = torch.concatenate(tag_feature_list, dim=0)
    embedded_img_feature_stack = torch.concatenate(embedded_img_feature_list, dim=0)
    embedded_tag_feature_stack = torch.concatenate(embedded_tag_feature_list, dim=0)
    return img_feature_stack, tag_feature_stack, embedded_img_feature_stack, embedded_tag_feature_stack

def extract_single_tag_feature(dataloder, emb_t):
    single_tag_feature_list = []
    embedded_single_tag_feature_list = []
    for data in dataloder:
        single_tag_feature = data.cuda(non_blocking=True)
        with torch.no_grad():
            embedded_single_tag_feature = emb_t(single_tag_feature)
        single_tag_feature_list.append(single_tag_feature)
        embedded_single_tag_feature_list.append(embedded_single_tag_feature)
    single_tag_feature_stack =  torch.concatenate(single_tag_feature_list, dim=0)
    embedded_single_tag_feature_stack = torch.concatenate(embedded_single_tag_feature_list, dim=0)
    return single_tag_feature_stack, embedded_single_tag_feature_stack


# define constant
params = utils.get_parameters()
BATCH_SIZE = params.batch_size
MODEL_PATH = params.model_path
IMG_FEATURE_PATH = params.img_feature_path
TAG_FEATURE_PATH = params.tag_feature_path
SINGLE_TAG_FEATURE_PATH = params.single_tag_feature_path
EMBEDDED_IMG_FEATURE_PATH = params.embedded_img_feature_path
EMBEDDED_TAG_FEATURE_PATH = params.embedded_tag_feature_path
EMBEDDED_SINGLE_TAG_FEATURE_PATH = params.embedded_single_tag_feature_path

# ディレクトリの作成
SAVE_DIR_IMG = os.path.dirname(EMBEDDED_IMG_FEATURE_PATH)
SAVE_DIR_TAG = os.path.dirname(EMBEDDED_TAG_FEATURE_PATH)
SAVE_DIR_SINGLE_TAG = os.path.dirname(EMBEDDED_SINGLE_TAG_FEATURE_PATH)
os.makedirs(SAVE_DIR_IMG, exist_ok=True)
os.makedirs(SAVE_DIR_TAG, exist_ok=True)
os.makedirs(SAVE_DIR_SINGLE_TAG, exist_ok=True)

# モデルの準備
device = torch.device('cuda:0')
emb_i = MLP.ReLU().to(device)
emb_t = MLP.ReLU().to(device)
emb_i.eval()
emb_t.eval()

# パラメータの読み込み
params = torch.load(MODEL_PATH)
emb_i.load_state_dict(params['emb_i'])
emb_t.load_state_dict(params['emb_t'])

# dataloderの準備
dataset = HierarchicalDataset.EvalDataset(IMG_FEATURE_PATH, TAG_FEATURE_PATH)
singletagset = HierarchicalDataset.SingleTagDataset(SINGLE_TAG_FEATURE_PATH)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
singletagloader = torch.utils.data.DataLoader(singletagset, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# 画像特徴，印象特徴(複数タグ)の保存
_, _, embedded_img_feature, embedded_tag_feature = extract_feature(dataloader, emb_i, emb_t)
torch.save(embedded_img_feature, EMBEDDED_IMG_FEATURE_PATH)
torch.save(embedded_tag_feature, EMBEDDED_TAG_FEATURE_PATH)

# 印象特徴(タグ単体)の保存
_, embedded_single_tag_feature = extract_single_tag_feature(singletagloader, emb_t)
torch.save(embedded_single_tag_feature, EMBEDDED_SINGLE_TAG_FEATURE_PATH)