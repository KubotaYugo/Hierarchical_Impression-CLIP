import torch
import numpy as np
from matplotlib import cm
from matplotlib.image import imsave

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


def retrieve_imp_img(img_features, tag_features, img_paths, tag_paths, direction, mode, k_max, save_dir):
    
    def metrics(query_tags, tags):
        '''
        タグの集合を比較する
        入力:   query_tags(検索のクエリとしたタグの集合)
                tag(検索によりピックアップされたタグ)
        出力:   precision, recall, f1-score
        '''
        true_positives = len(set(tags)&set(query_tags))
        rec_size = len(tags)
        act_size = len(query_tags)
        precision = true_positives/rec_size
        recall = true_positives/act_size
        f1_score = 2*(precision*recall)/(precision+recall) if (precision+recall)!=0 else 0.0
        precision  = format(precision, '.4f') 
        recall     = format(recall, '.4f')
        f1_score   = format(f1_score, '.4f')
        return precision, recall, f1_score

    # 保存用ディレクトリの作成
    temp_dir = f'{save_dir}/{direction}'
    os.makedirs(temp_dir, exist_ok=True)

    # 類似度行列を計算
    if direction=='imp2img':
        similarity_matrix = torch.matmul(img_features, tag_features.T).to('cpu').detach().numpy()
    elif direction=='img2imp':
        similarity_matrix = torch.matmul(img_features, tag_features.T).to('cpu').detach().numpy()
        similarity_matrix = similarity_matrix.T

    # top_kのインデックスを計算
    if mode=='upper':
        similarity_topk_args = np.argsort(-similarity_matrix, axis=0)
    elif mode=='lower':
        similarity_topk_args = np.argsort(similarity_matrix, axis=0)

    # 検索結果を保存
    for t in range(len(tag_features)):
        # クエリの画像をリストに追加
        query_img = utils.get_image_to_save(img_paths[t])
        output_img_list = [query_img]
        # クエリのタグをリストに追加
        font_name = os.path.basename(tag_paths[t])[:-4]
        similarity = format(similarity_matrix[t][t], '.4f')
        query_tags = utils.get_font_tags(tag_paths[t])
        precision, recall, f1_score = metrics(query_tags, query_tags)
        write_row = [font_name, similarity, precision, recall, f1_score]+query_tags
        write_row_list = [write_row]

        # キーの画像，タグをリストに追加
        for k in range(k_max):
            # キーの画像をリストに追加
            key_img = utils.get_image_to_save(img_paths[similarity_topk_args[k][t]])
            pad_v = np.ones(shape=(3, key_img.shape[1]))*255
            output_img_list.extend([pad_v, key_img])
            # キーのタグをリストに追加
            font_name = os.path.basename(tag_paths[similarity_topk_args[k][t]])[:-4]
            similarity = format(similarity_matrix[similarity_topk_args[k][t]][t], '.4f')
            key_tags = utils.get_font_tags(tag_paths[similarity_topk_args[k][t]])
            precision, recall, f1_score = metrics(query_tags, key_tags)
            write_row = [font_name, similarity, precision, recall, f1_score]+key_tags
            write_row_list.append(write_row)
        
        # 画像とcsvの保存
        query_font_name = os.path.basename(tag_paths[t])[:-4]
        output_img = np.vstack(output_img_list)
        imsave(f'{temp_dir}/{query_font_name}_{mode}.png', output_img, cmap=cm.gray)
        utils.save_list_to_csv(write_row_list, f'{temp_dir}/{query_font_name}_{mode}.csv')


def retrieve_tag2img(img_features, single_tag_features, img_paths, tag_paths, tag_list, mode, k_max, save_dir):
    # 保存用ディレクトリの作成
    temp_dir = f'{save_dir}/tag2img'
    os.makedirs(temp_dir, exist_ok=True)
    
    # 類似度行列とtop_kのインデックスを計算
    similarity_matrix = torch.matmul(img_features, single_tag_features.T).to('cpu').detach().numpy()
    if mode=='upper':
        similarity_topk_args = np.argsort(-similarity_matrix, axis=0)
    elif mode=='lower':
        similarity_topk_args = np.argsort(similarity_matrix, axis=0)

    # 検索結果を保存
    for t, query_tag in enumerate(tag_list):
        output_img_list = []
        write_row_list = []
        for k in range(k_max):        
            # キーの画像をリストに追加
            key_img = utils.get_image_to_save(img_paths[similarity_topk_args[k][t]])
            pad_v = np.ones(shape=(3, key_img.shape[1]))*255
            output_img_list.extend([pad_v, key_img])
            # キーのタグをリストに追加
            font_name = os.path.basename(img_paths[similarity_topk_args[k][t]])[:-4]
            similarity = format(similarity_matrix[similarity_topk_args[k][t]][t], '.4f')
            key_tags = utils.get_font_tags(tag_paths[similarity_topk_args[k][t]])
            flag = 1 if query_tag in key_tags else 0
            write_row = [flag, font_name, similarity]+key_tags
            write_row_list.append(write_row)
        # 画像とcsvの保存
        output_img = np.vstack(output_img_list)
        imsave(f'{temp_dir}/{query_tag}_{mode}.png', output_img, cmap=cm.gray)
        utils.save_list_to_csv(write_row_list, f'{temp_dir}/{query_tag}_{mode}.csv')


def retrieve_img2tag(img_features, single_tag_features, img_paths, tag_paths, tag_list, save_dir):
    # 保存用ディレクトリの作成
    temp_dir = f'{save_dir}/img2tag'
    os.makedirs(temp_dir, exist_ok=True)

    # 類似度行列とtop_kのインデックスを計算
    similarity_matrix = torch.matmul(img_features, single_tag_features.T).to('cpu').detach().numpy()
    similarity_matrix = similarity_matrix.T
    similarity_topk_args = np.argsort(-similarity_matrix, axis=0)
    
    # 検索結果を保存
    for f in range(len(img_paths)):
        query_tags = utils.get_font_tags(tag_paths[f])
        write_row_list = [query_tags]
        for k in range(len(tag_list)):
            # csvに保存する内容の整形
            key_tag = tag_list[similarity_topk_args[k][f]]
            flag = 1 if key_tag in query_tags else 0
            similarity = format(similarity_matrix[similarity_topk_args[k][f]][f], '.4f')
            write_row = [flag, key_tag, similarity]
            write_row_list.append(write_row)
        # csvの保存
        font_name = os.path.basename(img_paths[f])[:-4]
        utils.save_list_to_csv(write_row_list, f'{temp_dir}/{font_name}.csv')


# define constant
params = utils.get_parameters()
DATASET = params.dataset
BASE_DIR = params.base_dir
EMBEDDED_IMG_FEATURE_PATH = params.embedded_img_feature_path
EMBEDDED_TAG_FEATURE_PATH = params.embedded_tag_feature_path
EMBEDDED_SINGLE_TAG_FEATURE_PATH = params.embedded_single_tag_feature_path
SAVE_DIR = f'{BASE_DIR}/retrieval_result/{DATASET}'

# 特徴量の読み込み
embedded_img_feature = torch.load(EMBEDDED_IMG_FEATURE_PATH)
embedded_tag_feature = torch.load(EMBEDDED_TAG_FEATURE_PATH)
embedded_single_tag_feature = torch.load(EMBEDDED_SINGLE_TAG_FEATURE_PATH)

# 各種検索
K = 10
img_paths, tag_paths = utils.load_dataset_paths(DATASET)
tag_list = list(utils.get_tag_list())
retrieve_imp_img(embedded_img_feature, embedded_tag_feature, img_paths, tag_paths, 'imp2img', 'upper', K, SAVE_DIR)
retrieve_imp_img(embedded_img_feature, embedded_tag_feature, img_paths, tag_paths, 'imp2img', 'lower', K, SAVE_DIR)
retrieve_imp_img(embedded_img_feature, embedded_tag_feature, img_paths, tag_paths, 'img2imp', 'upper', K, SAVE_DIR)
retrieve_imp_img(embedded_img_feature, embedded_tag_feature, img_paths, tag_paths, 'img2imp', 'lower', K, SAVE_DIR)
retrieve_tag2img(embedded_img_feature, embedded_single_tag_feature, img_paths, tag_paths, tag_list, 'upper', K, SAVE_DIR)
retrieve_tag2img(embedded_img_feature, embedded_single_tag_feature, img_paths, tag_paths, tag_list, 'lower', K, SAVE_DIR)
retrieve_img2tag(embedded_img_feature, embedded_single_tag_feature, img_paths, tag_paths, tag_list, SAVE_DIR)