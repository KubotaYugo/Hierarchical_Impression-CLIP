"""
モデルの評価に用いる関数
"""
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import csv
import torch.nn.functional as F


class CustomDatasetForTag(Dataset):
    def __init__(self, tag_list, tokenizer):
        self.tag_list = tag_list
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.tag_list)
    def __getitem__(self, idx):
        tag = self.tag_list[idx]
        prompt = f"The impression is {tag}."
        tokenized_text = self.tokenizer(prompt, return_tensors="pt", max_length=self.tokenizer.max_model_input_sizes['openai/clip-vit-base-patch32'], padding="max_length", truncation=True)
        return tokenized_text['input_ids'][0]

def get_tag_list():
    """
    評価対象のタグをdictで返す
    """
    #　タグの取得(頻度順)
    with open(f"dataset/MyFonts_preprocessed/tag_freq_top10.csv") as f:
        reader = csv.reader(f)
        rows = np.asarray([row for row in reader])[1:]
    tag_freq = np.asarray(rows[:,2], dtype=np.int32)
    tag_list_org = rows[:,0][tag_freq>=50]
    # タグ番号→タグのdictを作成
    tag_list = {}
    for i, tag in enumerate(tag_list_org):
        tag_list[i] = tag
    return tag_list

def get_font_tags(tag_path):
    with open(tag_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        tags = [row for row in reader][0]
    return tags

def get_image_to_save(img_path, char=None):
    pad_h = np.ones(shape=(64, 1))*255
    img = np.load(img_path)["arr_0"].astype(np.float32)
    if char==None:
        images = img[0]
        for c in range(1,26):
            images = np.hstack([images, pad_h, img[c]])
    # elseで'ABC'を受け取ったらABC, 'HERONS'を受け取ったらHERONSを返すようにする
    return images

def extract_features(font_encoder, clip_model, dataloader):
    for idx, data in enumerate(dataloader):
        imgs, tokenized_tags = data
        imgs = imgs.cuda(non_blocking=True)
        tokenized_tags = tokenized_tags.cuda(non_blocking=True)
        with torch.no_grad():
            img_features = font_encoder(imgs)
            tag_features = clip_model.get_text_features(tokenized_tags)
            img_features = F.normalize(img_features, dim=1)
            tag_features = F.normalize(tag_features, dim=1) 
        if idx==0:
            img_features_stack = img_features
            tag_features_stack = tag_features
        else:
            img_features_stack = torch.concatenate((img_features_stack, img_features), dim=0)  
            tag_features_stack = torch.concatenate((tag_features_stack, tag_features), dim=0)
    return img_features_stack, tag_features_stack


def extract_text_features(dataloder, clip_model):
    with torch.no_grad():
        for i, data in enumerate(dataloder):
            tokenized_text = data.cuda(non_blocking=True)
            tag_features = clip_model.get_text_features(tokenized_text) 
            tag_features = F.normalize(tag_features, dim=1) 
            if i==0:
                tag_features_stack = tag_features
            else:
                tag_features_stack =  torch.concatenate((tag_features_stack, tag_features), dim=0)
    return tag_features


def retrieval_rank(similarity_matrix, mode=None):
    """
    類似度行列を受け取って, 印象から画像or画像から印象を検索したときのRetrievalRankを計算する 
    入力:
        similarity_matrix: 類似度行列
        mode:
            mode=="img2tag": 画像から印象を検索
            mode=="tag2img": 印象から画像を検索
    """
    if mode=="tag2img":
        similarity_matrix = similarity_matrix.T
    sorted_index = torch.argsort(similarity_matrix, dim=1, descending=True)
    rank = [torch.where(sorted_index[i]==i)[0].item()+1 for i in range(sorted_index.shape[0])]
    return rank


def get_font_tags(tag_path):
    with open(tag_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        tags = [row for row in reader][0]
    return tags

def AP_tag2img(embedded_img_features, embedded_single_tag_features, tag_list, tag_paths):
    similarity_matrix = torch.matmul(embedded_img_features, embedded_single_tag_features.T).to("cpu").detach().numpy()
    topk_args = np.argsort(-similarity_matrix, axis=0)
    AP = [0]*len(embedded_single_tag_features)
    for t in range(len(embedded_single_tag_features)):
        p = []
        count=0
        for k in range(len(embedded_img_features)):
            tags = get_font_tags(tag_paths[topk_args[k][t]])
            if tag_list[t] in tags:
                count += 1
                p.append(count/(k+1))
        AP[t]= np.sum(p)/count
    return AP

def AP_img2tag(embedded_img_features, embedded_single_tag_features, tag_list, tag_paths):
    similarity_matrix = torch.matmul(embedded_img_features, embedded_single_tag_features.T).to("cpu").detach().numpy()
    topk_args = np.argsort(-similarity_matrix, axis=1)
    AP = [0]*len(embedded_img_features)
    for f in range(len(embedded_img_features)):
        p = []
        query_tags = get_font_tags(tag_paths[f])
        count=0
        for k in range(len(embedded_single_tag_features)):
            if tag_list[topk_args[f][k]] in query_tags:
                count += 1
                p.append(count/(k+1))
        AP[f]= np.sum(p)/count
    return AP


def metrics(query_tags, tags):
    """
    タグの集合を比較する
    入力:   query_tags(検索のクエリとしたタグの集合)
            tag(検索によりピックアップされたタグ)
    出力:   precision, recall, f1-score
    """
    true_positives = len(set(tags)&set(query_tags))
    rec_size = len(tags)
    act_size = len(query_tags)
    precision = true_positives/rec_size
    recall = true_positives/act_size
    f1_score = 2*(precision*recall)/(precision+recall) if (precision+recall)!=0 else 0.0
    return precision, recall, f1_score