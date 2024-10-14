import os
from transformers import CLIPTokenizer, CLIPModel
import FontAutoencoder
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import utils
import eval_utils
import numpy as np


# define constant
EXP = utils.EXP
MODEL_PATH = f"{EXP}/model/best.pth.tar"
BATCH_SIZE = utils.BATCH_SIZE
DATASET = 'test'

# モデルの準備
device = torch.device('cuda:0')
font_autoencoder = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device)
font_encoder = font_autoencoder.encoder    
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# パラメータの読み込み
params = torch.load(MODEL_PATH)
font_encoder.load_state_dict(params['font_encoder'])
clip_model.load_state_dict(params['clip_model'])
font_encoder.eval()
clip_model.eval()

# dataloderの準備
img_paths, tag_paths = utils.LoadDatasetPaths(DATASET)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
dataset = utils.CustomDataset(img_paths, tag_paths, tokenizer)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# 単体タグのdataloder
tag_list = list(eval_utils.get_tag_list().values())
tagset = eval_utils.CustomDatasetForTag(tag_list, tokenizer)
tagloader = torch.utils.data.DataLoader(tagset, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)


# Average Retrieval Rankの計算
img_features, tag_features = eval_utils.extract_features(font_encoder, clip_model, dataloader)
similarity_matrix = torch.matmul(img_features, tag_features.T)
RR_img2tag = eval_utils.retrieval_rank(similarity_matrix, "img2tag")
RR_tag2img = eval_utils.retrieval_rank(similarity_matrix, "tag2img")
print(f"ARR_tag2img: {np.mean(RR_tag2img):.2f}")
print(f"ARR_img2tag: {np.mean(RR_img2tag):.2f}")

# mean Average Precisionの計算
single_tag_features = eval_utils.extract_text_features(tagloader, clip_model)
AP_tag2img = eval_utils.AP_tag2img(img_features, single_tag_features, tag_list, tag_paths)
AP_img2tag = eval_utils.AP_img2tag(tag_features, single_tag_features, tag_list, tag_paths)
print(f"AP_tag2img: {np.mean(AP_tag2img):.4f}")
print(f"AP_img2tag: {np.mean(AP_img2tag):.4f}")