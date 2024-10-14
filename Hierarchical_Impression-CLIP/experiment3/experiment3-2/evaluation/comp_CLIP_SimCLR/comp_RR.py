'''
CLIPとSimCLR(階層構造なし)の対応付け精度(Retrieval Rank)を比較する
CLIP:   ImpressionCLIP/experiment2/stop_by_ARR
SimCLR: Hierarchical_ImpressionCLIP/experiment3/experiment3-2
'''

from transformers import CLIPTokenizer, CLIPModel
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import matplotlib.pyplot as plt
import csv

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)
from models import FontAutoencoder
from models import MLP
from lib import utils
from lib import eval_utils


def culc_retrieval_rank(MODEL_PATH, loss):
    params = torch.load(MODEL_PATH)
    if loss=='CLIP':
        font_autoencoder.load_state_dict(params['font_encoder'])
    else:
        font_autoencoder.load_state_dict(params['font_autoencoder'])
    clip_model.load_state_dict(params['clip_model'])
    emb_i.load_state_dict(params['emb_i'])
    emb_t.load_state_dict(params['emb_t'])
    _, _, embedded_img_features, embedded_tag_features = eval_utils.extract_features(font_autoencoder, clip_model, emb_i, emb_t, dataloader)
    similarity_matrix = torch.matmul(embedded_img_features, embedded_tag_features.T)
    RR_tag2img = eval_utils.retrieval_rank(similarity_matrix, "tag2img")
    RR_img2tag = eval_utils.retrieval_rank(similarity_matrix, "img2tag")
    return RR_tag2img, RR_img2tag 


# define constant
DATASET = 'val'

# 保存用ディレクトリの作成
EXP = 'Hierarchical_Impression-CLIP/experiment3/experiment3-2'
LR = utils.LR
BATCH_SIZE = utils.BATCH_SIZE
SAVE_DIR = f'{EXP}/LR={LR}, BS={BATCH_SIZE}/comp_CLIP_SimCLR/comp_RR/{DATASET}'
os.makedirs(SAVE_DIR, exist_ok=True)


# モデルの準備
device = torch.device('cuda:0')
font_autoencoder = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
emb_i = MLP.ReLU().to(device)
emb_t = MLP.ReLU().to(device)
font_autoencoder.eval()
clip_model.eval()
emb_i.eval()
emb_t.eval()

# dataloderの準備
img_paths, tag_paths = utils.LoadDatasetPaths(DATASET)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
dataset = eval_utils.DMH_D_Eval(img_paths, tag_paths, tokenizer)


# Retrieval Rankの計算 (Impression-CLIP)
EXP = 'Impression-CLIP/experiment2/stop_by_ARR'
LR = 1e-4
BATCH_SIZE = 8192
MODEL_PATH = f'{EXP}/model/best.pth.tar'
dataloader = torch.utils.data.DataLoader(dataset, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
RR_tag2img_CLIP, RR_img2tag_CLIP = culc_retrieval_rank(MODEL_PATH, 'CLIP')

# Retrieval Rankの計算 (SimCLR(階層構造なし))
EXP = 'Hierarchical_Impression-CLIP/experiment4/experiment4-1'
LR = 1e-4
BATCH_SIZE = 32
MODEL_PATH = f'{EXP}/LR={LR}, BS={BATCH_SIZE}/results/model/best.pth.tar'
dataloader = torch.utils.data.DataLoader(dataset, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
RR_tag2img_SimCLR, RR_img2tag_SimCLR = culc_retrieval_rank(MODEL_PATH, 'SimCLR')


# 結果をcsvに保存 (retrieval_rank_tag2img)
fontnames = utils.get_fontnames(DATASET)
filename = f'{SAVE_DIR}/tag2img.csv'
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['fontname', 'diffrence', 'CLIP', 'SimCLR'])
    for i in range(len(img_paths)):    
        difference = RR_tag2img_CLIP[i] - RR_tag2img_SimCLR[i]
        writer.writerow([fontnames[i][0], difference, RR_tag2img_CLIP[i], RR_tag2img_SimCLR[i]])

# 結果をcsvに保存 (retrieval_rank_img2tag)
fontnames = utils.get_fontnames(DATASET)
filename = f'{SAVE_DIR}/img2tag.csv'
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['fontname', 'diffrence', 'CLIP', 'SimSLR'])
    for i in range(len(img_paths)): 
        difference = RR_img2tag_CLIP[i] - RR_img2tag_SimCLR[i]
        writer.writerow([fontnames[i][0], difference, RR_img2tag_CLIP[i], RR_img2tag_SimCLR[i]])