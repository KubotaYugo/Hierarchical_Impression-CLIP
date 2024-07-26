#------------------------------
#1つの印象語をクエリとした画像検索のAPを評価
#------------------------------
import torch
import torch.nn as nn
import torchvision
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.image import imread, imsave
from matplotlib import cm
from lib.dataloader import DataLoader
import lib.eval_utils as utils
import lib.models as models
import lib.tools as tools





K = 10
#----------保存用ディレクトリの作成----------
DIR_PATH = "20240126/jihun/result/co-training"
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



#----------画像特徴の埋め込み----------
img_features, _ = utils.EmbedImgImp(dataloder, model_img, model_imp, device)
 
#----------タグ特徴の埋め込み----------
imp_list = utils.GetImpList()
tag_features =  utils.EmbedTag(model_imp, imp_list)




font_paths, tag_paths = utils.LoadDatasetPaths(DATASET)
#----------各列でAP(Average Precision)を求める----------
AP = utils.CalcAP_Tag2Img(img_features, tag_features, imp_list, tag_paths)
#----------mAP----------
print(f"mAP={np.mean(AP)}")


#----------各タグのAPをcsvに保存----------
#-----trainとDATASETのデータ数を取得-----
with open(f"20240126/dataset/tag_freq_top10.csv") as f:
    reader = csv.reader(f)
    rows = np.asarray([row for row in reader])[1:]
tag_num_train = rows[:,2]
if DATASET=="train": tag_num = rows[:,2]
if DATASET=="val": tag_num = rows[:,3]
if DATASET=="test": tag_num = rows[:,4]
#-----csvに保存-----
rows = []
rows.append(["tag", "AP", "tag_num_train", f"tag_num_{DATASET}"])
for i in range(len(imp_list)):
    rows.append([imp_list[i], AP[i], tag_num_train[i], tag_num[i]])
rows = np.asarray(rows)
with open(f"{DIR_PATH}/evaluate/retrieve_tag2img_AP_{DATASET}.csv", 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(rows)