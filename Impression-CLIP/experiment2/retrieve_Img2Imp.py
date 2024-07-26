"""
画像をクエリ，キーを印象(testにある組み合わせで複数タグ)として検索
クエリとキーのcos類似度, タグのprecision, recall, f1, キーのタグをまとめたcsvとキーの画像を保存
"""
import torch
import os
import csv
import numpy as np
from matplotlib.image import imread, imsave
from matplotlib import cm

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
BATCH_SIZE = 8192
DATASET = "test"


#-----------保存用フォルダの準備----------
SAVE_PATH = f"{DIR_PATH}/Impression-CLIP/evaluate/retrieve_Img2Imp/{DATASET}"
os.makedirs(f"{SAVE_PATH}", exist_ok=True)

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




#----------フォントと印象をembedding----------
_, _, font_embedded_features, text_embedded_features = eval_utils.EmbedFontText(dataloader, font_autoencoder, clip_model, emb_f, emb_t, device)


#----------内積計算----------
#logits[i][j]は，font_embedded_features[i]とfont_embedded_features[f]の内積
logits = torch.matmul(font_embedded_features, text_embedded_features.T).to("cpu").detach().numpy()



#----------検索結果上位K個の印象とその画像の保存----------
logits_topk_args = np.argsort(-logits, axis=1)  #降順にするため，-logitsに
pad_h = np.ones(shape=(64, 1))*255
for f in range(len(font_embedded_features)):
    
    #----------クエリの印象が持つタグの取得-----------
    query_tags = eval_utils.GetFontTags(tag_paths[f])
    
    write_row = [[] for i in range(K)]
    for k in range(K):
        #----------キーの印象が持つタグの取得----------
        key_tags = eval_utils.GetFontTags(tag_paths[logits_topk_args[f][k]])
        #----------query_tagsとkey_tagsでprecision, recall, f1を計算----------
        precision, recall, f1_score = eval_utils.CulcSetMetrics(query_tags, key_tags)
        
        #----------csvに書く内容の整形----------
        font_name = os.path.basename(font_paths[logits_topk_args[f][k]])[:-4]
        similarity = format(logits[f][logits_topk_args[f][k]], ".4f")
        precision = format(precision, ".4f")
        recall = format(recall, ".4f")
        f1_score = format(f1_score, ".4f")
        write_row[k] = [font_name, similarity, precision, recall, f1_score]+key_tags
        
        #----------保存する画像の整形----------
        img = np.load(font_paths[logits_topk_args[f][k]])["arr_0"].astype(np.float32)
        input_images = img[0]
        for c in range(1,26):
            input_images = np.hstack([input_images, pad_h, img[c]])
        if k==0:
            output_images = input_images
        else:
            pad_v = np.ones(shape=(3, input_images.shape[1]))*255
            output_images = np.vstack([output_images, pad_v, input_images])
    
    #----------画像とcsvの保存----------
    font_name = os.path.basename(font_paths[f])[:-4]
    imsave(f"{SAVE_PATH}/{font_name}_upper.png", output_images, cmap=cm.gray)
    with open(f"{SAVE_PATH}/{font_name}_upper.csv", 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for w in range(K):
            writer.writerow(write_row[w])



#----------検索結果下位K個の印象とその画像の保存----------
logits_topk_args = np.argsort(logits, axis=1)  #昇順なのでマイナスなし
pad_h = np.ones(shape=(64, 1))*255
for f in range(len(text_embedded_features)):
    
    #----------クエリの印象が持つタグの取得-----------
    query_tags = eval_utils.GetFontTags(tag_paths[f])
        
    write_row = [[] for i in range(K)]
    for k in range(K):
        #----------キーの印象が持つタグの取得----------
        key_tags = eval_utils.GetFontTags(tag_paths[logits_topk_args[f][k]])
        #----------query_tagsとkey_tagsでprecision, recall, f1を計算----------
        precision, recall, f1_score = eval_utils.CulcSetMetrics(query_tags, key_tags)
        
        #----------csvに書く内容の整形----------
        font_name = os.path.basename(font_paths[logits_topk_args[f][k]])[:-4]
        similarity = format(logits[f][logits_topk_args[f][k]], ".4f")
        precision = format(precision, ".4f")
        recall = format(recall, ".4f")
        f1_score = format(f1_score, ".4f")
        write_row[(K-1-k)] = [font_name, similarity, precision, recall, f1_score]+key_tags
        
        #----------保存する画像の整形----------
        img = np.load(font_paths[logits_topk_args[f][k]])["arr_0"].astype(np.float32)
        input_images = img[0]
        for c in range(1,26):
            input_images = np.hstack([input_images, pad_h, img[c]])
        if k==0:
            output_images = input_images
        else:
            pad_v = np.ones(shape=(3, input_images.shape[1]))*255
            output_images = np.vstack([input_images, pad_v, output_images])
    
    #----------画像とcsvの保存----------
    font_name = os.path.basename(font_paths[f])[:-4]
    imsave(f"{SAVE_PATH}/{font_name}_lower.png", output_images, cmap=cm.gray)
    with open(f"{SAVE_PATH}/{font_name}_lower.csv", 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for w in range(K):
            writer.writerow(write_row[w])
