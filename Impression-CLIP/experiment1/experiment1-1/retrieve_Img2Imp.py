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
import csv
from matplotlib.image import imread, imsave
from matplotlib import cm


# define constant
EXP = "experiment1"
MODEL_PATH = f"Impression-CLIP/{EXP}/model/best.pth.tar"
BATCH_SIZE = 1024
DATASET = 'test'
K = 10
SAVE_PATH = f"Impression-CLIP/{EXP}/retrieve_img2imp/{DATASET}"
os.makedirs(SAVE_PATH, exist_ok=True)

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

# 特徴量の抽出
img_features, tag_features = eval_utils.extract_features(font_encoder, clip_model, dataloader)
similarity_matrix = torch.matmul(img_features, tag_features.T).to("cpu").detach().numpy()



# 検索結果上位K個の印象とその画像の保存
similarity_matrix_topk_args = np.argsort(-similarity_matrix, axis=1)  #降順にするため，-similarity_matrixに
pad_h = np.ones(shape=(64, 1))*255
for f in range(len(img_features)):
    
    # クエリの印象が持つタグの取得
    query_tags = eval_utils.get_font_tags(tag_paths[f])
    
    write_row = [[] for i in range(K)]
    for k in range(K):
        # キーの印象が持つタグの取得
        key_tags = eval_utils.get_font_tags(tag_paths[similarity_matrix_topk_args[f][k]])
        # query_tagsとkey_tagsでprecision, recall, f1を計算
        precision, recall, f1_score = eval_utils.metrics(query_tags, key_tags)
        
        # csvに書く内容の整形
        font_name = os.path.basename(img_paths[similarity_matrix_topk_args[f][k]])[:-4]
        similarity = format(similarity_matrix[f][similarity_matrix_topk_args[f][k]], ".4f")
        precision = format(precision, ".4f")
        recall = format(recall, ".4f")
        f1_score = format(f1_score, ".4f")
        write_row[k] = [font_name, similarity, precision, recall, f1_score]+key_tags
        
        # 保存する画像の整形
        img = eval_utils.get_image_to_save(img_paths[similarity_matrix_topk_args[f][k]])
        if k==0:
            output_images = img
        else:
            pad_v = np.ones(shape=(3, img.shape[1]))*255
            output_images = np.vstack([output_images, pad_v, img])
    
    # 画像とcsvの保存
    font_name = os.path.basename(img_paths[f])[:-4]
    imsave(f"{SAVE_PATH}/{font_name}_upper.png", output_images, cmap=cm.gray)
    with open(f"{SAVE_PATH}/{font_name}_upper.csv", 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for w in range(K):
            writer.writerow(write_row[w])



# 検索結果下位K個の印象とその画像の保存
similarity_matrix_topk_args = np.argsort(similarity_matrix, axis=1)  #昇順なのでマイナスなし
pad_h = np.ones(shape=(64, 1))*255
for f in range(len(img_features)):
    # クエリの印象が持つタグの取得-
    query_tags = eval_utils.get_font_tags(tag_paths[f])
        
    write_row = [[] for i in range(K)]
    for k in range(K):
        # キーの印象が持つタグの取得
        key_tags = eval_utils.get_font_tags(tag_paths[similarity_matrix_topk_args[f][k]])
        # query_tagsとkey_tagsでprecision, recall, f1を計算
        precision, recall, f1_score = eval_utils.metrics(query_tags, key_tags)
        
        # csvに書く内容の整形
        font_name = os.path.basename(img_paths[similarity_matrix_topk_args[f][k]])[:-4]
        similarity = format(similarity_matrix[f][similarity_matrix_topk_args[f][k]], ".4f")
        precision = format(precision, ".4f")
        recall = format(recall, ".4f")
        f1_score = format(f1_score, ".4f")
        write_row[(K-1-k)] = [font_name, similarity, precision, recall, f1_score]+key_tags
        
        # 保存する画像の整形
        img = eval_utils.get_image_to_save(img_paths[similarity_matrix_topk_args[f][k]])
        if k==0:
            output_images = img
        else:
            pad_v = np.ones(shape=(3, img.shape[1]))*255
            output_images = np.vstack([img, pad_v, output_images])
    
    # 画像とcsvの保存
    font_name = os.path.basename(img_paths[f])[:-4]
    imsave(f"{SAVE_PATH}/{font_name}_lower.png", output_images, cmap=cm.gray)
    with open(f"{SAVE_PATH}/{font_name}_lower.csv", 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for w in range(K):
            writer.writerow(write_row[w])
