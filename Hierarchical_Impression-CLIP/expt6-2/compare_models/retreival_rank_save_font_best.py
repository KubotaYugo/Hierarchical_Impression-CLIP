'''
baselineとaverage, iterative, label_andについてそれぞれベストなモデルで比較し, 
RR_tag2imgとRR_imgtagの双方で良くなったものtop10と悪くなったものtop10のフォントを保存する
'''

import torch
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from matplotlib import cm
import seaborn as sns

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils
from lib import eval_utils


def get_RR(params, param_set):
    # パラメータ設定
    params.loss_type = param_set[0]
    params.weights = param_set[1]
    params.random_seed = param_set[2]

    params.base_dir = f'{params.expt}/co-embedding/C=[{params.num_img_clusters}, {params.num_tag_clusters}]_{params.tag_preprocess}_{params.temperature}_{params.learn_temperature}_{params.initial_temperature}_{params.loss_type}_{params.ce_bce}_W={params.weights}_seed={params.random_seed}'
    params.embedded_img_feature_path = f'{params.base_dir}/feature/embedded_img_feature/{params.dataset}.pth'
    params.embedded_tag_feature_path = f'{params.base_dir}/feature/embedded_tag_feature/{params.dataset}.pth'
    EMBEDDED_IMG_FEATURE_PATH = params.embedded_img_feature_path
    EMBEDDED_TAG_FEATURE_PATH = params.embedded_tag_feature_path

    # 特徴量の読み込み
    embedded_img_feature = torch.load(EMBEDDED_IMG_FEATURE_PATH)
    embedded_tag_feature = torch.load(EMBEDDED_TAG_FEATURE_PATH)

    # Retrieval Rankの計算
    similarity_matrix = torch.matmul(embedded_img_feature, embedded_tag_feature.T)
    RR_tag2img = eval_utils.retrieval_rank(similarity_matrix, 'tag2img')
    RR_img2tag = eval_utils.retrieval_rank(similarity_matrix, 'img2tag')

    return {'tag2img':RR_tag2img, 'img2tag':RR_img2tag}

def save_fonts(indexes, order):
    for i in range(comparison.shape[0]):
        output_img_list = []
        write_row_list = []
        for index in indexes[i]:
            # 保存する画像の読み込み & マージ
            img = utils.get_image_to_save(img_paths[index])
            pad_v = np.ones(shape=(60, img.shape[1]))*255
            output_img_list.extend([pad_v, img])

            # csvに保存する内容の読み込み
            fontname = Path(img_paths[index]).stem
            tag = utils.get_font_tags(tag_paths[index])
            img_cluster = img_cluster_id[index]
            tag_cluster = tag_cluster_id[index]
            RR_tag2img_baseline = RR_list['tag2img'][0][index]
            RR_tag2img_comparison = RR_list['tag2img'][i+1][index]
            RR_tag2img_difference = RR_tag2img_comparison-RR_tag2img_baseline
            RR_img2tag_baseline = RR_list['img2tag'][0][index]
            RR_img2tag_comparison = RR_list['img2tag'][i+1][index]
            RR_img2tag_difference = RR_img2tag_comparison-RR_img2tag_baseline
            write_row = [fontname, img_cluster, tag_cluster, 
                         RR_tag2img_baseline, RR_tag2img_comparison, RR_tag2img_difference,
                         RR_img2tag_baseline, RR_img2tag_comparison, RR_img2tag_difference] + tag
            write_row_list.append(write_row)
                
        # 画像とcsvの保存
        output_img = np.vstack(output_img_list)
        filename = f'{SAVE_DIR}/{model_name[0]}_{model_name[i+1]}_{direction}_{order}{K}'
        imsave(f'{filename}.png', output_img, cmap=cm.gray)
        utils.save_list_to_csv(write_row_list, f'{filename}.csv')


def tag_statistics(indexes, order):
    for i in range(comparison.shape[0]):
        # タグの頻度を取得
        tags_list = [utils.get_font_tags(tag_paths[index]) for index in indexes[i]]
        tag_frequency = Counter(tag for tags in tags_list for tag in tags)
        tag_frequency_dict = {tag:0 for tag in entire_tag_list}
        for key_tag in tag_frequency.keys():
            tag_frequency_dict[key_tag] = tag_frequency[key_tag]

        # タグの頻度をプロット
        fig, ax = plt.subplots(figsize=(24, 6))
        plt.bar(tag_frequency_dict.keys(), tag_frequency_dict.values(), edgecolor='black', width=1)
        plt.xticks(fontsize=5.3, rotation=90)
        plt.xlabel(f'Tag')
        plt.ylabel('Frequency') 
        filename = f'{SAVE_DIR}/{model_name[0]}_{model_name[i+1]}_{direction}_{order}{K}'
        plt.savefig(f'{filename}_tag_freq.png', dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()

        # DATASET全体に対する割合を計算
        entire_tag_ratio_dict = {tag:0 for tag in entire_tag_list}
        entire_tag_freq = Counter(tag for tag_path in tag_paths for tag in utils.get_font_tags(tag_path))
        for key_tag in tag_frequency.keys():
            entire_tag_ratio_dict[key_tag] = tag_frequency_dict[key_tag]/entire_tag_freq[key_tag]
        # タグのDATASET全体に対する割合をプロット
        fig, ax = plt.subplots(figsize=(24, 6))
        plt.bar(entire_tag_ratio_dict.keys(), entire_tag_ratio_dict.values(), edgecolor='black', width=1)
        plt.xticks(fontsize=5.3, rotation=90)
        plt.xlabel(f'Tag')
        plt.ylabel('Frequency')
        plt.savefig(f'{filename}_tag_freq_ratio.png', dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()

        # タグの個数の頻度をプロット
        number_of_tags = [len(tags) for tags in tags_list]
        number_of_tags_freq = Counter(number_of_tags)
        number_of_tags_dict = {i+1:0 for i in range(10)}
        for key_num in number_of_tags_freq:
            number_of_tags_dict[key_num] = number_of_tags_freq[key_num]
        plt.bar(number_of_tags_dict.keys(), number_of_tags_dict.values(), edgecolor='black', width=1)
        plt.xticks(ticks=[x+1 for x in range(10)], labels=number_of_tags_dict.keys())
        plt.xlim(0, 10+1)
        plt.xlabel(f'Number of tags')
        plt.ylabel('Frequency')
        plt.savefig(f'{filename}_number_of_tags.png', dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()

        # クラスタの頻度の保存
        img_cluster_id_i = img_cluster_id[indexes[i]]
        tag_cluster_id_i = tag_cluster_id[indexes[i]]
        pairs = list(zip(img_cluster_id_i, tag_cluster_id_i))   # 同じインデックスの要素のペアを作成
        pair_count = Counter(pairs)                             # ペアの頻度をカウント
        heatmap = np.zeros((10, 10))                            # ペアの頻度をヒートマップ用の配列に反映
        for (i, j), count in pair_count.items():
            heatmap[i][j] = count
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap, annot=True, cmap="YlGnBu", fmt='g', cbar=True)
        plt.xlabel("tag cluster id")
        plt.ylabel("img cluster id")
        plt.savefig(f'{filename}_cluster_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()


# define constant
params = utils.get_parameters()
EXPT = params.expt
DATASET = params.dataset
BASE_DIR = params.base_dir
IMG_CLUSTER_PATH = params.img_cluster_path
TAG_CLUSTER_PATH = params.tag_cluster_path

# ラベル(クラスタID)の取得
img_cluster_id = np.load(IMG_CLUSTER_PATH)['arr_0'].astype(np.int64)
tag_cluster_id = np.load(TAG_CLUSTER_PATH)['arr_0'].astype(np.int64)

# loss_type, weights
param_set = [['average',   [1.0, 0.0, 0.0,], 1],
             ['average',   [1.0, 1.0, 1.0,], 3],
             ['iterative', [1.0, 1.0, 1.0,], 5],
             ['label_and', [1.0, 1.0, 1.0,], 5]]
model_name = ['baseline', 'average', 'iterative', 'label_and']

# ディレクトリの作成
K = 50
SAVE_DIR = f'{EXPT}/compare_models/retrieval_rank_save_font_best/{DATASET}/K={K}'
os.makedirs(SAVE_DIR, exist_ok=True)

# retrieval rankの計算
# RR_list_org[i]: param_set[i]のRR_tag2imgとRR_img2tag
RR_list_org = [get_RR(params, param_set_i) for param_set_i in param_set]

# ベストなモデルの検索順位を取得
RR_list = {'tag2img':[], 'img2tag':[]}
for direction in ['tag2img', 'img2tag']:
    for i in range(len(param_set)):
        RR_list[direction].append(RR_list_org[i][direction])
    RR_list[direction] = np.asarray(RR_list[direction])

# ラベル(クラスタID), 画像, 印象タグのパスの取得
entire_tag_list = utils.get_tag_list() # 全体のタグのリスト(trainの頻度順)
_, tag_paths = utils.load_dataset_paths(DATASET)
img_cluster_id = np.load(IMG_CLUSTER_PATH)['arr_0'].astype(np.int64)
tag_cluster_id = np.load(TAG_CLUSTER_PATH)['arr_0'].astype(np.int64)
img_paths, tag_paths = utils.load_dataset_paths(DATASET)

for direction in ['tag2img', 'img2tag']:
    # baselineとそれぞれのモデルを比較し，良くなったものtopK, bottomKのインデックスを取得
    RR_list_direction = np.asarray(RR_list[direction])
    baseline = RR_list_direction[0]
    comparison = RR_list_direction[1:]
    difference = comparison-baseline    # 大きいと悪くなってる
    topk_index = np.argsort(difference, axis=1)[:, :K] 
    bottomk_index = np.argsort(difference, axis=1)[:, -K:]

    # topK, bottomKの，画像，フォント名，検索順位，タグを保存
    save_fonts(topk_index, 'top')
    save_fonts(bottomk_index, 'bottom')

    # タグの統計情報を保存
    tag_statistics(topk_index, 'top')
    tag_statistics(bottomk_index, 'bottom')