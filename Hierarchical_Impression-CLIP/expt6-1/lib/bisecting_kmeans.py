import numpy as np
from sklearn.cluster import KMeans


def calculate_within_cluster_variance(cluster):
    center = np.mean(cluster, axis=0)
    variance = np.sum((cluster - center) ** 2)
    return variance


def bisecting_kmeans(data, num_clusters):
    clusters = [data]
    variances = [calculate_within_cluster_variance(data)]
    data_index = [[i for i in range(len(data))]]
    
    while len(clusters) < num_clusters: 
        # 分散最大のクラスターの情報を取りだす
        largest_variance_idx = np.argmax(variances)
        largest_cluster = clusters.pop(largest_variance_idx)
        largest_variance = variances.pop(largest_variance_idx)
        largest_index = data_index.pop(largest_variance_idx)
        
        kmeans = KMeans(n_clusters=2, n_init=1).fit(largest_cluster)
        labels = kmeans.labels_

        # 特徴量を新規クラスタで分割
        cluster_1 = largest_cluster[labels==0]
        cluster_2 = largest_cluster[labels==1]
        clusters.append(cluster_1)
        clusters.append(cluster_2)
        
        # indexを新規クラスタで分割
        data_index_1  = np.asarray(largest_index)[labels==0]
        data_index_2  = np.asarray(largest_index)[labels==1]
        data_index.append(data_index_1)
        data_index.append(data_index_2)
        
        variances.append(calculate_within_cluster_variance(cluster_1))
        variances.append(calculate_within_cluster_variance(cluster_2))

    # data_indexの形式を変換
    trasformed_data_index = np.zeros(len(data))
    for i in range(len(data_index)):
        trasformed_data_index[data_index[i]] = i     

    return clusters, trasformed_data_index


def calculate_inertia(X, labels):
    """
    Calculate the inertia for given data points and cluster labels.
    
    Parameters:
    - X: numpy array of shape (n_samples, n_features), feature matrix
    - labels: numpy array of shape (n_samples,), cluster labels
    
    Returns:
    - inertia: float, sum of squared distances to nearest cluster centers
    """
    inertia = 0.0
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        cluster_points = X[labels == label]                     # 現在のクラスタに属するデータポイントを抽出
        centroid = np.mean(cluster_points, axis=0)              # クラスタの重心を計算
        inertia += np.sum((cluster_points - centroid) ** 2)     # データポイントと重心の距離の二乗を計算し、合計

    return inertia


def replace_label(label):
    '''
    [2, 2, 1, 4, 5, 3, 2, 5, 4] -> [0, 0, 1, 2, 3, 4, 0, 3, 2]
    のように，リスト先頭から出てくる順に番号を振り直す
    '''
    unique_numbers = {}
    replaced_label = []
    current_number = 0
    for num in label:
        if num not in unique_numbers:
            unique_numbers[num] = current_number
            current_number += 1
        replaced_label.append(unique_numbers[num])
    return replaced_label