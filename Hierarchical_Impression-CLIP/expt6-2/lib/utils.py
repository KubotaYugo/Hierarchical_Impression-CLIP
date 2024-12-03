import torch
import numpy as np
import csv
import random
import argparse
from dotmap import DotMap


def get_parameters():
    params = {
        'expt':                     'Hierarchical_Impression-CLIP/expt6-2',
        'fontautoencoder_path':     'FontAutoencoder/model/best.pt',
        'max_epoch':                 10000,     # 固定
        'early_stopping_patience':   100,       # 固定
        'learning_rate':             [1e-4, 1e-4, 1e-4], # 固定  emb_img, emb_tag, temperatureの学習率
        'batch_size':                8192,      # 固定
        'num_img_clusters':          10,        # いったん固定
        'num_tag_clusters':          10,        # いったん固定
        'tag_preprocess':           ['normal', 'average_single_tag', 'average_upto_10'][1],
        'temperature':              ['ExpMultiplier', 'ExpMultiplierLogit'][1],
        'learn_temperature':        [True, False][0],
        'initial_temperature':      [0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0][4],
        'loss_type':                ['average', 'iterative', 'label_and'][0],
        'ce_bce':                   ['CE', 'BCE'][1],
        'weights':                  [1.0, 1.0, 1.0],    # WEIGHT_PAIR, WEIGHT_IMG, WEIGHT_TAG
        'random_seed':              [1, 2, 3, 4, 5][0],
        'dataset':                  ['train', 'val', 'test'][0],
    }
    params = DotMap(params)

    parser = argparse.ArgumentParser()
    parser.add_argument('--expt', type=str)
    parser.add_argument('--fontautoencoder_path', type=str)
    parser.add_argument('--max_epoch', type=int)
    parser.add_argument('--early_stopping_patience', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_img_clusters', type=int)
    parser.add_argument('--num_tag_clusters', type=int)
    parser.add_argument('--tag_preprocess', type=str)
    parser.add_argument('--temperature', type=str)
    parser.add_argument('--learn_temperature', type=bool)
    parser.add_argument('--initial_temperature', type=int)
    parser.add_argument('--loss_type', type=str)
    parser.add_argument('--ce_bce', type=str)
    parser.add_argument('--weights', type=float, nargs='+')
    parser.add_argument('--random_seed', type=int)
    parser.add_argument('--dataset', type=str)
    
    args = parser.parse_args()
    for key, value in vars(args).items():
        if value is not None:
            params[key] = value

    params.base_dir = f'{params.expt}/co-embedding/C=[{params.num_img_clusters}, {params.num_tag_clusters}]_{params.tag_preprocess}_{params.temperature}_{params.learn_temperature}_{params.initial_temperature}_{params.loss_type}_{params.ce_bce}_W={params.weights}_seed={params.random_seed}'
    params.model_path = f'{params.base_dir}/results/model/best.pth.tar'
    
    params.img_feature_path = f'{params.expt}/feature/img_feature/{params.dataset}.pth'
    params.tag_feature_path = f'{params.expt}/feature/tag_feature/{params.tag_preprocess}/{params.dataset}.pth'
    params.single_tag_feature_path = f'{params.expt}/feature/tag_feature/single_tag/{params.dataset}.pth'
    
    params.img_cluster_path = f'{params.expt}/clustering/cluster/img/{params.dataset}/{params.num_img_clusters}.npz'
    params.tag_cluster_path = f'{params.expt}/clustering/cluster/tag/{params.tag_preprocess}/{params.dataset}/{params.num_tag_clusters}.npz'
    
    params.embedded_img_feature_path = f'{params.base_dir}/feature/embedded_img_feature/{params.dataset}.pth'
    params.embedded_tag_feature_path = f'{params.base_dir}/feature/embedded_tag_feature/{params.dataset}.pth'
    params.embedded_single_tag_feature_path = f'{params.base_dir}/feature/embedded_single_tag_feature/{params.dataset}.pth'

    # for key, value in params.items():
    #     print(f"{key}: {value}")
    # print('------------------------------')

    return params


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm


def load_dataset_paths(dataset):
    """
    datasetのフォントとタグのパスのリストを返す
    """
    with open(f"dataset/MyFonts_preprocessed/tag_txt/fontname/{dataset}.csv") as f:
        reader = csv.reader(f)
        font_names = np.asarray([row for row in reader])
    image_paths = [f"dataset/MyFonts_preprocessed/font_numpy_Impression-CLIP/{dataset}/{font_name[0]}.npz" for font_name in font_names]
    tag_paths = [f"dataset/MyFonts_preprocessed/tag_txt/{dataset}/{font_name[0]}.csv" for font_name in font_names]
    return image_paths, tag_paths


def get_fontnames(dataset):
    with open(f"dataset/MyFonts_preprocessed/tag_txt/fontname/{dataset}.csv") as f:
        reader = csv.reader(f)
        font_names = np.asarray([row[0] for row in reader])
    return font_names

def get_font_tags(tag_path):
    with open(tag_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        tags = [row for row in reader][0]
    return tags

def get_tag_list():
    with open(f"dataset/MyFonts_preprocessed/tag_freq_top10.csv") as f:
        reader = csv.reader(f)
        rows = np.asarray([row for row in reader])[1:]
    tag_freq = np.asarray(rows[:,2], dtype=np.int32)
    tag_list = rows[:,0][tag_freq>=50]
    return tag_list

def save_list_to_csv(data_list, output_path):
    '''
    リストの各要素を1行としてCSVファイルに保存する。
    :param data_list: リスト (各要素が1行として保存される)
    :param output_path: 出力CSVファイルのパス
    '''
    with open(output_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_list)