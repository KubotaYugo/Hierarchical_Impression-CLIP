import numpy as np
import torch
import csv
from matplotlib.image import imread, imsave
from matplotlib import cm

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import eval_utils


def retrieve_imp_img(img_features, tag_features, img_paths, tag_paths, direction, mode, k_max, save_dir):
    # calcurate topk args
    if direction=='imp2img':
        similarity_matrix = torch.matmul(img_features, tag_features.T).to("cpu").detach().numpy()
    elif direction=='img2imp':
        similarity_matrix = torch.matmul(img_features, tag_features.T).to("cpu").detach().numpy()
        similarity_matrix = similarity_matrix.T

    if mode=='upper':
        similarity_topk_args = np.argsort(-similarity_matrix, axis=0)
    elif mode=='lower':
        similarity_topk_args = np.argsort(similarity_matrix, axis=0)

    os.makedirs(f'{save_dir}/retrieve_{direction}', exist_ok=True)
    for t in range(len(tag_features)):
        # クエリの印象が持つタグの取得
        query_tags = eval_utils.get_font_tags(tag_paths[t])
        
        write_rows = [[] for i in range(k_max)]
        for k in range(k_max):
            # キーのフォントが持つタグの取得
            key_tags = eval_utils.get_font_tags(tag_paths[similarity_topk_args[k][t]])
            # query_tagsとkey_tagsでprecision, recall, f1を計算
            precision, recall, f1_score = eval_utils.metrics(query_tags, key_tags)
            
            # csvに書く内容の整形
            font_name = os.path.basename(img_paths[similarity_topk_args[k][t]])[:-4]
            similarity = format(similarity_matrix[similarity_topk_args[k][t]][t], ".4f")
            precision = format(precision, ".4f")
            recall = format(recall, ".4f")
            f1_score = format(f1_score, ".4f")
            if mode=='upper':
                write_rows[k] = [font_name, similarity, precision, recall, f1_score]+key_tags
            elif mode=='lower':
                write_rows[(k_max-1-k)] = [font_name, similarity, precision, recall, f1_score]+key_tags
            
            # 保存する画像の整形
            img = eval_utils.get_image_to_save(img_paths[similarity_topk_args[k][t]])
            if k==0:
                output_images = img
            else:
                pad_v = np.ones(shape=(3, img.shape[1]))*255
                if mode=='upper':
                    output_images = np.vstack([output_images, pad_v, img])
                elif mode=='lower':
                    output_images = np.vstack([img, pad_v, output_images])
        
        # 画像とcsvの保存
        font_name = os.path.basename(img_paths[t])[:-4]
        imsave(f"{save_dir}/retrieve_{direction}/{font_name}_{mode}.png", output_images, cmap=cm.gray)
        with open(f"{save_dir}/retrieve_{direction}/{font_name}_{mode}.csv", 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(write_rows)


def retrieve_tag2img(img_features, single_tag_features, img_paths, tag_paths, tag_list, mode, k_max, save_dir):
    # calcurate topk args
    similarity_matrix = torch.matmul(img_features, single_tag_features.T).to("cpu").detach().numpy()
    if mode=='upper':
        similarity_topk_args = np.argsort(-similarity_matrix, axis=0)
    elif mode=='lower':
        similarity_topk_args = np.argsort(similarity_matrix, axis=0)

    os.makedirs(f'{save_dir}/retrieve_tag2img', exist_ok=True)
    for t in range(len(tag_list)):
        write_rows = [[] for i in range(k_max)]
        for k in range(k_max):        
            # フォントが持つタグの取得
            # フォントが持つタグに, クエリが入っていればflag=1
            flag = 0
            tags = eval_utils.get_font_tags(tag_paths[similarity_topk_args[k][t]])
            if tag_list[t] in tags:
                flag = 1

            # csvに書く内容の整形
            font_name = os.path.basename(img_paths[similarity_topk_args[k][t]])[:-4]
            similarity = format(similarity_matrix[similarity_topk_args[k][t]][t], ".4f")
            if mode=='upper':
                write_rows[k] = [flag, font_name, similarity]+tags
            elif mode=='lower':
                write_rows[(k_max-1)-k] = [flag, font_name, similarity]+tags
            
            # 保存する画像の整形
            img = eval_utils.get_image_to_save(img_paths[similarity_topk_args[k][t]])
            if k==0:
                output_images = img
            else:
                pad_v = np.ones(shape=(3, img.shape[1]))*255
                if mode=='upper':
                    output_images = np.vstack([output_images, pad_v, img])
                elif mode=='lower':
                    output_images = np.vstack([img, pad_v, output_images])

        # 画像とcsvの保存
        imsave(f"{save_dir}/retrieve_tag2img/{tag_list[t]}_{mode}.png", output_images, cmap=cm.gray)
        with open(f"{save_dir}/retrieve_tag2img/{tag_list[t]}_{mode}.csv", 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(write_rows)


def retrieve_img2tag(img_features, single_tag_features, img_paths, tag_paths, tag_list, save_dir):
    # calcurate topk args
    similarity_matrix = torch.matmul(img_features, single_tag_features.T).to("cpu").detach().numpy()
    similarity_topk_args = np.argsort(-similarity_matrix, axis=1)  # 降順にするため，-similarityに
    
    os.makedirs(f'{save_dir}/retrieve_img2tag', exist_ok=True)
    for f in range(len(img_paths)):
        # クエリのフォントが持つタグの取得
        query_tags = eval_utils.get_font_tags(tag_paths[f])
        
        write_rows = [[] for i in range(len(tag_list))]
        for k in range(len(tag_list)):
            # 近傍のタグがクエリに入っていればflag=1
            flag = 0
            if tag_list[similarity_topk_args[f][k]] in query_tags:
                flag = 1
            # csvに書く内容の整形
            similarity = format(similarity_matrix[f][similarity_topk_args[f][k]], ".4f")
            write_rows[k] = [flag, tag_list[similarity_topk_args[f][k]], similarity]
        
        # csvの保存
        font_name = os.path.basename(img_paths[f])[:-4]
        with open(f"{save_dir}/retrieve_img2tag/{font_name}.csv", 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(write_rows)