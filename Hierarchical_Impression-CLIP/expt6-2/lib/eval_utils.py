import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
import csv
from matplotlib import cm
from matplotlib.image import imsave

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils



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
    rank = (sorted_index==torch.arange(sorted_index.shape[0]).unsqueeze(1).to('cuda')).nonzero(as_tuple=True)[1] + 1
    return rank.tolist()

def retrieval_rank_matrix(similarity_matrix, mode=None):
    """
    類似度行列を受け取って, 印象から画像or画像から印象を検索したときのRetrievalRankを計算する 
    入力:
        similarity_matrix: 類似度行列
        mode:
            mode=="img2tag": 画像から印象を検索
            mode=="tag2img": 印象から画像を検索
    sorted_index[i][j]は, データiをクエリとして検索したときのj番目の検索結果のインデックス
    """
    if mode=="tag2img":
        similarity_matrix = similarity_matrix.T
    sorted_index = torch.argsort(similarity_matrix, dim=1, descending=True)
    return sorted_index

def AP_tag2img(embedded_img_features, embedded_single_tag_features, tag_list, tag_paths):
    similarity_matrix = torch.matmul(embedded_img_features, embedded_single_tag_features.T).to("cpu").detach().numpy()
    topk_args = np.argsort(-similarity_matrix, axis=0)
    AP = []
    all_tags = [utils.get_font_tags(tag_paths[i]) for i in range(len(tag_paths))]
    for t, query_tag in enumerate(tag_list):
        # Get the top-k indices for the current tag
        top_indices = topk_args[:, t]
        # Extract the tags corresponding to these indices
        key_tags = [all_tags[i] for i in top_indices]
        # Create a boolean mask for whether the target tag is in the relevant tags
        match_mask = np.array([query_tag in tags for tags in key_tags])
        # Compute precision at each rank
        ranks = np.arange(1, len(match_mask) + 1)
        precision_at_k = np.cumsum(match_mask) / ranks
        # Filter precision values where there is a match
        precision_at_relevant = precision_at_k[match_mask]
        # Compute Average Precision (AP)
        AP.append(np.mean(precision_at_relevant))
    return AP

def AP_img2tag(embedded_img_features, embedded_single_tag_features, tag_list, tag_paths):
    similarity_matrix = torch.matmul(embedded_img_features, embedded_single_tag_features.T).to("cpu").detach().numpy()
    topk_args = np.argsort(-similarity_matrix, axis=1)
    all_tags = [set(utils.get_font_tags(tag_paths[f])) for f in range(len(tag_paths))]
    AP = []
    for f, query_tags in enumerate(all_tags):
        # Get the sorted tag indices for the current image
        sorted_tag_indices = topk_args[f]
        # Create a boolean mask for relevant tags
        relevant_mask = [tag_list[idx] in query_tags for idx in sorted_tag_indices]
        # Compute precision at each rank
        ranks = np.arange(1, len(relevant_mask) + 1)
        precision_at_k = np.cumsum(relevant_mask) / ranks
        # Filter precision values where relevant tags exist
        precision_at_relevant = precision_at_k[relevant_mask]
        # Compute Average Precision (AP) for the current image
        AP.append(np.mean(precision_at_relevant))
    return AP

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


# def AP_tag2img(embedded_img_features, embedded_single_tag_features, tag_list, tag_paths):
#     similarity_matrix = torch.matmul(embedded_img_features, embedded_single_tag_features.T).to("cpu").detach().numpy()
#     topk_args = np.argsort(-similarity_matrix, axis=0)
#     AP = [0]*len(embedded_single_tag_features)
#     for t in range(len(embedded_single_tag_features)):
#         p = []
#         count=0
#         for k in range(len(embedded_img_features)):
#             tags = utils.get_font_tags(tag_paths[topk_args[k][t]])
#             if  tag_list[t] in tags:
#                 count += 1
#                 p.append(count/(k+1))
#         AP[t]= np.sum(p)/count
#     return AP

# def AP_img2tag(embedded_img_features, embedded_single_tag_features, tag_list, tag_paths):
#     similarity_matrix = torch.matmul(embedded_img_features, embedded_single_tag_features.T).to("cpu").detach().numpy()
#     topk_args = np.argsort(-similarity_matrix, axis=1)
#     AP = [0]*len(embedded_img_features)
#     for f in range(len(embedded_img_features)):
#         p = []
#         query_tags = utils.get_font_tags(tag_paths[f])
#         count=0
#         for k in range(len(embedded_single_tag_features)):
#             if tag_list[topk_args[f][k]] in query_tags:
#                 count += 1
#                 p.append(count/(k+1))
#         AP[f]= np.sum(p)/count
#     return AP