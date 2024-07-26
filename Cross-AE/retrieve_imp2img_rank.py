"""
印象から画像を検索したときの対応するペアの順位をまとめたCSVを出力
印象から画像を検索したときの対応するペアの順位の分布をグラフ化
"""
import torch
import torch.nn as nn
import torchvision
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
from lib.dataloader import DataLoader
import lib.eval_utils as utils
import lib.models as models
import lib.tools as tools





#----------保存用ディレクトリの作成----------
DIR_PATH = "20240126/jihun/result/co-training/"
os.makedirs(f"{DIR_PATH}/evaluate", exist_ok=True)



#----------dataloder----------
DATASET = "test"
img_dir = f"20240126/dataset/font_jihun/{DATASET}"
label_dir = "20240126/dataset/imp_jihun"
dataset = tools.FontDataloader2(img_dir, label_dir)
NUM_WORKERS = 0     #ここを0にしないと動かないかも(環境による？)
dataloder = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=NUM_WORKERS)



#----------モデル, オプティマイザ, ロスの準備----------
device = 'cuda'
#-----img-AE-----
resnet18 = torchvision.models.resnet18([2, 2, 2, 2], weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
resnet18.conv1 = nn.Conv2d(26, 64, kernel_size=7, stride=2, padding=3,bias=False)
resnet_without_fc = nn.Sequential(*list(resnet18.children())[:-2])
additional_conv_layer = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0,bias=False)
custom_resnet = nn.Sequential(resnet_without_fc, additional_conv_layer)
model_img = models.Image_AE_model(custom_resnet, models.Decoder()).to(device)
model_path = '20240126/jihun/result/co-training/img-AE/best(epoch=8).pt'
state_dict = torch.load(model_path, map_location='cuda:0')
model_img.load_state_dict(state_dict)
model_img.eval()
#-----imp-AE-----
model_imp = models.IntegratedModel(models.DeepSets(), models.imp_Encoder(), models.imp_Decoder()).to(device)
model_path = '20240126/jihun/result/co-training/imp-AE/best(epoch=8).pt'
state_dict = torch.load(model_path, map_location='cuda:0')
model_imp.load_state_dict(state_dict)
model_imp.eval()




img_features, imp_features = utils.EmbedImgImp(dataloder, model_img, model_imp, device)
rank_matrix, _ = utils.RankMatrix(imp_features, img_features)

sorted_index = np.argsort(rank_matrix, axis=1)
rank = [np.where(sorted_index[i]==i)[0].item()+1 for i in range(sorted_index.shape[0])]



#----------csvに保存----------
font_names = utils.GetFontnames(DATASET)
with open(f"{DIR_PATH}/evaluate/retrieve_imp2img_rank_{DATASET}.csv", 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for i in range(len(font_names)):
        writer.writerow([font_names[i], rank[i]])

#----------順位の分布をプロット----------
fig, ax = plt.subplots(figsize=(7, 4.8))
bin_width = 10
bins = np.arange(1, max(rank)+bin_width, bin_width)
plt.hist(rank, bins=bins, edgecolor='black')
plt.xlim(0, 1750)
plt.ylim(0, 200)
plt.savefig(f"{DIR_PATH}/evaluate/retrieve_imp2img_rank_{DATASET}.png", dpi=300, bbox_inches='tight')


print(f"Average retrieval rank(imp2img) = {np.mean(rank)}")