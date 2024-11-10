import torch
import numpy as np
import csv
import random
import argparse
import itertools

def get_parameters():
    params = {
        'expt':                     'Hierarchical_Impression-CLIP/expt6-1',
        'fontautoencoder_path':     'FontAutoencoder/model/best.pt',
        'max_epoch':                 10000,
        'early_stopping_patience':   50,
        'num_img_clusters':          10,
        'num_tag_clusters':          10,
        'learning_rate':             10000,
        'batch_size':                8192,
        'weights':                  [1.0, 1.0, 1.0],
        'dataset':                  ['train', 'val', 'test'][2],
        'tag_preprocess':           ['normal', 'average_single_tag', 'average_upto_10', 'single_tag'][0]
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--expt', type=str)
    parser.add_argument('--fontautoencoder_path', type=str)
    parser.add_argument('--max_epoch', type=int)
    parser.add_argument('--early_stopping_patience', type=int)
    parser.add_argument('--num_img_clusters', type=float)
    parser.add_argument('--num_tag_clusters', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--weights', type=float, nargs='+')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--tag_preprocess', type=str)
    
    args = parser.parse_args()
    for key, value in vars(args).items():
        if value is not None:
            params[key] = value

    params['base_dir'] = f'{params['expt']}/LR={params['learning_rate']}_BS={params['batch_size']}_C=[{params['num_img_clusters']}, {params['num_tag_clusters']}]_W={params['weights']}_{params['tag_preprocess']}'
    params['model_path'] = f'{params['base_dir']}/results/model/best.pth.tar'
    params['img_feature_path'] = f'{params['expt']}/img_features/{params['dataset']}.pth'
    params['tag_feature_path'] = f'{params['expt']}/tag_features/{params['dataset']}/{params['tag_preprocess']}.pth'
    params['img_cluster_path'] = f'{params['expt']}/clustering/img/{params['dataset']}/{params['num_img_clusters']}.npz'
    params['tag_cluster_path'] = f'{params['expt']}/clustering/tag/{params['dataset']}/{params['tag_preprocess']}/{params['num_tag_clusters']}.npz'

    for key, value in params.items():
        print(f"{key}: {value}")
    print('------------------------------')

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

def padding_tags(tags_org, max_length=10):
    tags = tags_org
    while len(tags)+len(tags_org)<=max_length:
        tags = tags+tags_org
    all_combinations = itertools.product(tags_org, repeat=max_length-len(tags))
    add = [list(p) for p in all_combinations if len(set(p)) == len(p)]
    if add!=['']:
        return_list = [tags+a for a in add]
    else:
        return_list = tags
    return return_list

# def get_tag_list():
#     """
#     評価対象のタグをdictで返す
#     """
#     #　タグの取得(頻度順)
#     with open(f"dataset/MyFonts_preprocessed/tag_freq_top10.csv") as f:
#         reader = csv.reader(f)
#         rows = np.asarray([row for row in reader])[1:]
#     tag_freq = np.asarray(rows[:,2], dtype=np.int32)
#     tag_list_org = rows[:,0][tag_freq>=50]
#     # タグ番号→タグのdictを作成
#     tag_list = {}
#     for i, tag in enumerate(tag_list_org):
#         tag_list[i] = tag
#     return tag_list