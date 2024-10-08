import os
from transformers import CLIPTokenizer, CLIPModel
import FontAutoencoder
import MLP
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import utils
import eval_utils
import retrieve_utils
import numpy as np


# define constant
EXP = "experiment3"
MODEL_PATH = f"Hierarchical_Impression-CLIP/{EXP}/results/model/best.pth.tar"
BATCH_SIZE = 256
DATASET = 'test'
SAVE_PATH = f'Hierarchical_Impression-CLIP/{EXP}/retrieve'

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


# パラメータの読み込み
params = torch.load(MODEL_PATH)
font_autoencoder.load_state_dict(params['font_autoencoder'])
clip_model.load_state_dict(params['clip_model'])
emb_i.load_state_dict(params['emb_i'])
emb_t.load_state_dict(params['emb_t'])

# dataloderの準備
img_paths, tag_paths = utils.LoadDatasetPaths(DATASET)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
dataset = eval_utils.DMH_D_Eval(img_paths, tag_paths, tokenizer)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
# 単体タグのdataloder
tag_list = list(eval_utils.get_tag_list().values())
tagset = eval_utils.DMH_D_ForTag(tag_list, tokenizer)
tagloader = torch.utils.data.DataLoader(tagset, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# 特徴量の抽出
_, _, embedded_img_features, embedded_tag_features = eval_utils.extract_features(font_autoencoder, clip_model, emb_i, emb_t, dataloader)
_, embedded_single_tag_features = eval_utils.extract_text_features(tagloader, clip_model, emb_t)

# 各種検索
retrieve_utils.retrieve_imp_img(embedded_img_features, embedded_tag_features, img_paths, tag_paths, 'imp2img', 'upper', 10, SAVE_PATH, DATASET)
retrieve_utils.retrieve_imp_img(embedded_img_features, embedded_tag_features, img_paths, tag_paths, 'imp2img', 'lower', 10, SAVE_PATH, DATASET)
retrieve_utils.retrieve_imp_img(embedded_img_features, embedded_tag_features, img_paths, tag_paths, 'img2imp', 'upper', 10, SAVE_PATH, DATASET)
retrieve_utils.retrieve_imp_img(embedded_img_features, embedded_tag_features, img_paths, tag_paths, 'img2imp', 'lower', 10, SAVE_PATH, DATASET)
retrieve_utils.retrieve_tag2img(embedded_img_features, embedded_single_tag_features, img_paths, tag_paths, tag_list, 'upper', 10, SAVE_PATH, DATASET)
retrieve_utils.retrieve_tag2img(embedded_img_features, embedded_single_tag_features, img_paths, tag_paths, tag_list, 'lower', 10, SAVE_PATH, DATASET)
retrieve_utils.retrieve_img2tag(embedded_img_features, embedded_single_tag_features, img_paths, tag_paths, tag_list, SAVE_PATH, DATASET)