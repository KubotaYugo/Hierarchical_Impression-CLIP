import torch
import numpy as np
import csv
import random
import argparse
from dotmap import DotMap


def get_parameters():
    params = {
        'expt':                     'Hierarchical_Impression-CLIP/expt6-2-2',
        'fontautoencoder_path':     'FontAutoencoder/model/best.pt',
        'max_epoch':                 10000,                 # 固定
        'early_stopping_patience':   500,                   # 固定
        'learning_rate':             [1e-4, 1e-4, 1e-4],    # 固定  emb_img, emb_tag, temperatureの学習率
        'batch_size':                8192,                  # 固定
        'tag_preprocess':           'average_single_tag',   # 固定
        'initial_temperature':       0.15,                  # 固定 (固定でいいのか？)
        'loss_type':                ['SupCon', 'BCE'][0],
        'weights':                  [1.0, 1.0],             # w_img2tag, w_tag2img 
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

    params.base_dir = f'{params.expt}/co-embedding/{params.loss_type}_W={params.weights}_BS={params.batch_size}_seed={params.random_seed}'
    params.model_path = f'{params.base_dir}/results/model/best.pth.tar'

    params.img_feature_path = f'{params.expt}/feature/img_feature/{params.dataset}.pth'
    params.tag_feature_path = f'{params.expt}/feature/tag_feature/{params.tag_preprocess}/{params.dataset}.pth'
    params.single_tag_feature_path = f'{params.expt}/feature/tag_feature/single_tag/{params.dataset}.pth'
    
    params.img_cluster_path = f'{params.expt}/clustering/clusterID/img/{params.dataset}.npz'
    params.tag_cluster_path = f'{params.expt}/clustering/clusterID/tag/{params.tag_preprocess}/{params.dataset}.npz'
    
    params.embedded_img_feature_path = f'{params.base_dir}/feature/embedded_img_feature/{params.dataset}.pth'
    params.embedded_tag_feature_path = f'{params.base_dir}/feature/embedded_tag_feature/{params.dataset}.pth'
    params.embedded_single_tag_feature_path = f'{params.base_dir}/feature/embedded_single_tag_feature/{params.dataset}.pth'

    # クラスタの学習を入れずに学習した場合 (ベースライン)
    params.base_dir_baseline = 'Hierarchical_Impression-CLIP/expt6-2/co-embedding/C=[10, 10]_average_single_tag_ExpMultiplierLogit_True_0.15_average_BCE_W=[1.0, 0.0, 0.0]_seed=1'
    params.model_path_baseline = f'{params.base_dir_baseline}/results/model/best.pth.tar'
    params.embedded_img_feature_path_baseline = f'{params.base_dir_baseline}/feature/embedded_img_feature/{params.dataset}.pth'
    params.embedded_tag_feature_path_baseline = f'{params.base_dir_baseline}/feature/embedded_tag_feature/{params.dataset}.pth'
    params.embedded_single_tag_feature_path_baseline = f'{params.base_dir_baseline}/feature/embedded_single_tag_feature/{params.dataset}.pth'

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

def load_hierarchical_clusterID(cluster_path, depth=None):
    '''
    指定した深さまでの階層的なクラスタのラベルを返す
    '''
    cluster_id = np.load(cluster_path)["arr_0"].astype(np.int64)
    cluster_id_prune = cluster_id if depth is None else cluster_id[:, :depth]
    cluster_id_marge = [''.join(map(str, row)) for row in cluster_id_prune]
    return np.asarray(cluster_id_marge)


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
    # trainの頻度順になってる
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
    with open(output_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data_list)

def get_image_to_save(img_path, char=None):
    pad_h = np.ones(shape=(64, 1))*255
    img = np.load(img_path)['arr_0'].astype(np.float32)
    if char==None:
        images = img[0]
        for c in range(1,26):
            images = np.hstack([images, pad_h, img[c]])
    return images