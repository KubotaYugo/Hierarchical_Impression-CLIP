import warnings
import numpy as np
from sklearn.cluster import KMeans

import sys
sys.setrecursionlimit(10000)


def recursive_kmeans(indexes, features, current_depth=0):
    
    if len(indexes) <= 1:
        return {'depth': current_depth, 'index': indexes, 'feature': features}

    # KMeansを使って2分割
    with warnings.catch_warnings(record=True) as W:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(features)
        
        # 同じ特徴量は同じクラスタに
        for warn in W:
            if isinstance(warn.message, UserWarning) and \
            "Number of distinct clusters (1) found smaller than n_clusters (2)" in str(warn.message):
                return {'depth': current_depth, 'index': indexes, 'feature': features}
            
    # クラスタごとにデータを分ける
    clusters = {i: features[labels == i] for i in range(2)}
    cluster_indexes = {i: indexes[labels == i] for i in range(2)}

    # 再帰的に分割
    return {
                'depth':  current_depth,
                'index':  indexes,
                0: recursive_kmeans(cluster_indexes[0], clusters[0], current_depth + 1),
                1: recursive_kmeans(cluster_indexes[1], clusters[1], current_depth + 1)
            }


def get_cluster_path(cluster_tree, data_index, path=[]):
    '''
    与えられたインデックスまでのパスを取得
    '''
    if 'feature' in cluster_tree:
        return path
    if data_index in cluster_tree[0]['index']:
        return get_cluster_path(cluster_tree[0], data_index, path + [0])
    else:
        return get_cluster_path(cluster_tree[1], data_index, path + [1])


def print_clusters(cluster_tree, indent=0):
    '''
    クラスタの構造を表示
    '''
    prefix = '   ' * indent
    if 'feature' in cluster_tree:
        print(f'{prefix}Leaf cluster at depth {cluster_tree['depth']}, index: {cluster_tree['index']}, ')
    else:
        print(f'{prefix}Node at depth {cluster_tree['depth']}, index: {cluster_tree['index']}')
        print_clusters(cluster_tree[0], indent + 2)
        print_clusters(cluster_tree[1], indent + 2)


def pad_array(data):
    # パディングする最大長を計算
    max_len = max(len(row) for row in data)

    # NumPy 配列を作成し、2 で埋める
    padded_array = np.full((len(data), max_len), fill_value=2, dtype=int)

    # 各行を埋め込む
    for i, row in enumerate(data):
        padded_array[i, :len(row)] = row
    return padded_array