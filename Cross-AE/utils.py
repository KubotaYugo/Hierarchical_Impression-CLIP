import torch
import numpy as np
import glob
import csv


def extract_features(dataloder, model_img, model_imp, device):
    with torch.no_grad():
        for i, data in enumerate(dataloder):    
            image, w2v, _ = data["image"], data["w2v"], data["w2v_mean"]
            image = image.to(device)
            # 画像
            img_feature = model_img.encoder(image)
            # 印象側
            _, tag_feature, _ = model_imp(w2v)
            # concatenate
            if i==0:
                img_features = img_feature
                tag_features = tag_feature
            else:
                img_features =  torch.concatenate((img_features, img_feature), dim=0)
                tag_features =  torch.concatenate((tag_features, tag_feature), dim=0)
    img_features = img_features.to('cpu').detach().numpy().copy()
    tag_features = tag_features.to('cpu').detach().numpy().copy()
    return img_features, tag_features


def rank_matrix(query, key):
    query = np.asarray(query, dtype=np.float64)
    key = np.asarray(key, dtype=np.float64)
    distance_matrix = np.zeros((len(query), len(key)))
    for i in range(len(query)):
        distance_matrix[i,:] = np.linalg.norm(query[i]-key, ord=2, axis=1)
    nearest_idx = np.argsort(distance_matrix, axis=1)
    rank_matrix = np.zeros_like(nearest_idx)
    rows, cols = np.indices(nearest_idx.shape)
    rank_matrix[rows, nearest_idx] = cols
    return rank_matrix, distance_matrix