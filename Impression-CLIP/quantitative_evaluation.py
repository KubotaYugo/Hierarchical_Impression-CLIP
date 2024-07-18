import torch
import os
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
BATCH_SIZE = 8192
DATASET = "test"



#-----------保存用フォルダの準備----------
SAVE_PATH = f"{DIR_PATH}/Impression-CLIP/evaluate"
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


#----------全フォント・印象をembedding----------
font_features, text_features, font_embedded_features, text_embedded_features = eval_utils.EmbedFontText(dataloader, font_autoencoder, clip_model, emb_f, emb_t, device)
#----------内積計算----------
#logits[i][j]は，font_embeded_features[i]とtext_embeded_features[f]の内積
similarity_matrix = torch.matmul(font_embedded_features, text_embedded_features.T)



#----------AverageRetrievalRank(Imp2Img, Img2Imp)の計算----------
AverageRetrievalRank_Imp2Img = np.mean(eval_utils.CalcRetrievalRank(similarity_matrix, mode="Imp2Img"))
AverageRetrievalRank_Img2Imp = np.mean(eval_utils.CalcRetrievalRank(similarity_matrix, mode="Img2Imp"))

#----------mAP(Tag2Img, Img2Tag)の計算----------
#-----タグ単体のdataloderの準備----
tag_list = list(eval_utils.GetTagList().values())    #評価対象とするタグのリスト
tagset = eval_utils.DatasetForTag(tag_list, clip.tokenize)
tagloader = torch.utils.data.DataLoader(tagset, batch_size=256, shuffle=False, num_workers = os.cpu_count(), pin_memory=True)
#----------全タグをenocde&embed----------
_, tag_embedded_features = eval_utils.EmbedText(tagloader, clip_model, emb_t, device)
mAP_Tag2Img = np.mean(eval_utils.CalcAP_Tag2Img(font_embedded_features, tag_embedded_features, tag_list, tag_paths))
mAP_Img2Tag = np.mean(eval_utils.CalcAP_Img2Tag(font_embedded_features, tag_embedded_features, tag_list, tag_paths))


print(f"AverageRetrievalRank_Imp2Img: {AverageRetrievalRank_Imp2Img:.1f}")
print(f"AverageRetrievalRank_Img2Imp: {AverageRetrievalRank_Img2Imp:.1f}")
print(f"mAP_Tag2Img: {mAP_Tag2Img:.4f}")
print(f"mAP_Img2Tag: {mAP_Img2Tag:.4f}")