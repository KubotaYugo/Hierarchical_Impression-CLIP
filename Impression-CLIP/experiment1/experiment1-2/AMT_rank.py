"""
AMT-setを使って, 人の目で見て最もその印象語らしいものを正しく上位にピックアップできるか
"""
import torch
import os
import csv
import numpy as np

import clip
import utils
import eval_utils

from pathlib import Path
DIR_PATH = Path(__file__).resolve().parent.parent.parent # /ICDAR_Kubotaまでのパスを取得

import sys
sys.path.append(f"{Path(__file__).resolve().parent.parent}")
import FontAutoencoder.FontAutoencoder as FontAutoencoder


#---------ハイパラ----------
K = 10
EPOCH = 10000
BATCH_SIZE = 8192
LEARNING_RATE = 1e-4
DATASET = "test"


#----------モデルの準備----------
device = torch.device('cuda:0')
font_autoencoder = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device)    
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
emb_f = utils.MLP().to(device)
emb_t = utils.MLP().to(device)
#-----パラメータのロード-----
font_autoencoder_state, clip_model_state, emb_f_state, emb_t_state, temp_state = list(torch.load(f"{DIR_PATH}/Impression-CLIP/model/best.pt").values())
font_autoencoder.load_state_dict(font_autoencoder_state)
clip_model.load_state_dict(clip_model_state)
emb_f.load_state_dict(emb_f_state)
emb_t.load_state_dict(emb_t_state)
#-----評価モードに-----
font_autoencoder.eval()
clip_model.eval()
emb_f.eval()
emb_t.eval()


#----------データの準備----------
font_paths, tag_paths = utils.LoadDatasetPaths(DATASET)
dataset = utils.CustomDataset(font_paths, tag_paths, clip.tokenize)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers = os.cpu_count(), pin_memory=True)

#-----タグ単体のdataloderの準備----
tag_list = list(eval_utils.GetTagList().values())    #評価対象とするタグのリスト
tagset = eval_utils.DatasetForTag(tag_list, clip.tokenize)
tagloader = torch.utils.data.DataLoader(tagset, batch_size=256, shuffle=False, num_workers = os.cpu_count(), pin_memory=True)




#----------フォントをembedding----------
_, _, font_embedded_features, _ = eval_utils.EmbedFontText(dataloader, font_autoencoder, clip_model, emb_f, emb_t, device)
#----------タグをembedding----------
_, tag_embedded_features = eval_utils.EmbedText(tagloader, clip_model, emb_t, device)

#----------内積計算----------
similarity_matrix = torch.matmul(font_embedded_features, tag_embedded_features.T).to("cpu").detach().numpy()



#----------AMT-setの読み込み-----------
with open(f"{DIR_PATH}/dataset/preprocess/AMT-set.csv", encoding='utf8', newline='') as f:
    csvreader = csv.reader(f, delimiter=',')
    rows = np.asarray([row for row in csvreader])[1:]
AMT_tags = rows[:,0]
AMT_fonts = [rows[i,1:4] for i in range(len(rows))]
AMT_labels = np.asarray(rows[:,4], dtype=np.int32)



#----------タグ名→タグの番号に変換するdict----------
tag_num_dict = {}
for i in range(len(tag_list)):
    tag_num_dict[tag_list[i]] = i
#----------フォント名→フォントの番号に変換するdict----------
font_num_dict = {}
for i in range(len(font_paths)):
    font_num_dict[os.path.basename(font_paths[i])[:-4]] = i


number = []
accuracy_flag = []
retrieval_rank = []
sim_list = []
evaluated_tags = []
evaluated_labels = []
for i in range(len(AMT_tags)):
    if AMT_tags[i] in tag_list:
        tag_num = tag_num_dict[AMT_tags[i]]
        font_num = [font_num_dict[AMT_fonts[i][k]] for k in range(3)]
        similarlity = [similarity_matrix[font_num[k]][tag_num] for k in range(3)]
        sim_list.append(similarlity)

        if similarlity[AMT_labels[i]] == np.max(similarlity):
            accuracy_flag.append(1)
        else:
            accuracy_flag.append(0)
            
        number.append(i)
        evaluated_tags.append(AMT_tags[i])
        evaluated_labels.append(AMT_labels[i])
        retrieval_rank.append(np.where(np.argsort(similarlity)[::-1]==AMT_labels[i])[0][0]+1)
        
print(f"accuracy={np.mean(accuracy_flag):.4f}")
print(f"average retrieval rank={np.mean(retrieval_rank):.3f}")


with open(f"{DIR_PATH}/Impression-CLIP/evaluate/AMT_set.csv", 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=',')
    for i in range(len(evaluated_tags)):
        writer.writerow([number[i], evaluated_tags[i], accuracy_flag[i], retrieval_rank[i], evaluated_labels[i], sim_list[i][0], sim_list[i][1], sim_list[i][2]])