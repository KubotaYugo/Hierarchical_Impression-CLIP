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


# define constant
EXP = "experiment1"
MODEL_PATH = f"Impression-CLIP/{EXP}/model/best.pth.tar"
BATCH_SIZE = 1024
DATASET = 'test'
K = 10
SAVE_PATH = f"Impression-CLIP/{EXP}/retrieve_img2tag/{DATASET}"
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

# 単体タグのdataloder
tag_list = list(eval_utils.get_tag_list().values())
tagset = eval_utils.CustomDatasetForTag(tag_list, tokenizer)
tagloader = torch.utils.data.DataLoader(tagset, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# 特徴量の抽出
img_features, _ = eval_utils.extract_features(font_encoder, clip_model, dataloader)
single_tag_features = eval_utils.extract_text_features(tagloader, clip_model)
similarity_matrix = torch.matmul(img_features, single_tag_features.T).to("cpu").detach().numpy()


# 検索結果上位K個のフォントとそのフォントが持つタグの保存
logits_topk_args = np.argsort(-similarity_matrix, axis=1)  # 降順にするため，-logitsに
pad_h = np.ones(shape=(64, 1))*255
for f in range(len(img_paths)):
    # クエリのフォントが持つタグの取得
    query_tags = eval_utils.get_font_tags(tag_paths[f])
    
    write_rows = [[] for i in range(len(tag_list))]
    for k in range(len(tag_list)):
        # 近傍のタグがクエリに入っていればflag=1
        flag = 0
        if tag_list[logits_topk_args[f][k]] in query_tags:
            flag = 1
        # csvに書く内容の整形
        similarity = format(similarity_matrix[f][logits_topk_args[f][k]], ".4f")
        write_rows[k] = [flag, tag_list[logits_topk_args[f][k]], similarity]
    
    # csvの保存
    font_name = os.path.basename(img_paths[f])[:-4]
    with open(f"{SAVE_PATH}/{font_name}.csv", 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(write_rows)