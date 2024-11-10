import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
import csv
from matplotlib import cm
from matplotlib.image import imread, imsave

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils




def get_image_to_save(img_path, char=None):
    pad_h = np.ones(shape=(64, 1))*255
    img = np.load(img_path)["arr_0"].astype(np.float32)
    if char==None:
        images = img[0]
        for c in range(1,26):
            images = np.hstack([images, pad_h, img[c]])
    # メモ: elseで'ABC'を受け取ったらABC, 'HERONS'を受け取ったらHERONSを返すようにしたい
    return images


def save_fonts(fontlist, dataset, filename):
    # 画像を保存
    imgs_hstacks = []
    pad_h = np.ones(shape=(64, 1))*255
    pad_v = np.ones(shape=(3, 64*26+1*25))*255
    for fontname in fontlist:
        fontpath = f'dataset/MyFonts_preprocessed/font_numpy_Impression-CLIP/{dataset}/{fontname}.npz' 
        imgs = np.load(fontpath)["arr_0"].astype(np.float32)
        imgs_hstacks.append(np.hstack([imgs[0]]+[np.hstack([pad_h, imgs[i]]) for i in range(1, 26)]))
    output_img = np.vstack([imgs_hstacks[0]]+[np.vstack([pad_v, imgs_hstacks[i]]) for i in range(1, len(imgs_hstacks))])
    imsave(f'{filename}.png', output_img, cmap=cm.gray)

    # タグを保存
    write_rows = [['fontname', 'tags']]
    for fontname in fontlist:
        tag_path = f'dataset/MyFonts_preprocessed/tag_txt/{dataset}/{fontname}.csv'
        tags = utils.get_font_tags(tag_path)
        write_rows.append([fontname]+tags)
    with open(f'{filename}.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(write_rows)


def extract_features(font_autoencoder, clip_model, emb_i, emb_t, dataloader):
    for idx, data in enumerate(dataloader):
        imgs, tokenized_tags = data
        imgs = imgs.cuda(non_blocking=True)
        tokenized_tags = tokenized_tags.cuda(non_blocking=True)
        with torch.no_grad():
            img_features = font_autoencoder.encoder(imgs)
            tag_features = clip_model.get_text_features(tokenized_tags) 
            embedded_img_features = emb_i(img_features)
            embedded_tag_features = emb_t(tag_features)
        if idx==0:
            img_features_stack = img_features
            tag_features_stack = tag_features
            embedded_img_features_stack = embedded_img_features
            embedded_tag_features_stack = embedded_tag_features
        else:
            img_features_stack = torch.concatenate((img_features_stack, img_features), dim=0)  
            tag_features_stack = torch.concatenate((tag_features_stack, tag_features), dim=0)
            embedded_img_features_stack = torch.concatenate((embedded_img_features_stack, embedded_img_features), dim=0)
            embedded_tag_features_stack = torch.concatenate((embedded_tag_features_stack, embedded_tag_features), dim=0)
    return img_features_stack, tag_features_stack, embedded_img_features_stack, embedded_tag_features_stack


def extract_text_features(dataloder, clip_model, emb_t):
    with torch.no_grad():
        for i, data in enumerate(dataloder):
            tokenized_text = data.cuda(non_blocking=True)
            tag_feature = clip_model.get_text_features(tokenized_text) 
            embedded_tag_feature = emb_t(tag_feature.to(dtype=torch.float32))
            if i==0:
                tag_features = tag_feature
                embedded_tag_features = embedded_tag_feature
            else:
                tag_features =  torch.concatenate((tag_features, tag_feature), dim=0)
                embedded_tag_features = torch.concatenate((embedded_tag_features, embedded_tag_feature), dim=0)
    return tag_features, embedded_tag_features


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

def AP_tag2img(embedded_img_features, embedded_single_tag_features, tag_list, tag_paths):
    similarity_matrix = torch.matmul(embedded_img_features, embedded_single_tag_features.T).to("cpu").detach().numpy()
    topk_args = np.argsort(-similarity_matrix, axis=0)
    AP = [0]*len(embedded_single_tag_features)
    for t in range(len(embedded_single_tag_features)):
        p = []
        count=0
        for k in range(len(embedded_img_features)):
            tags = utils.get_font_tags(tag_paths[topk_args[k][t]])
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
        query_tags = utils.get_font_tags(tag_paths[f])
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