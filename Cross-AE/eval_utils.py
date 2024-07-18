import torch
import numpy as np
import glob
import csv


def EmbedImgImp(dataloder, model_img, model_imp, device):
    """
    画像と印象の中間表現を計算
    """
    with torch.no_grad():
        for i, data in enumerate(dataloder):    
            image, w2v, _ = data["image"], data["w2v"], data["w2v_mean"]
            #-----画像側-----
            image = image.to(device)
            grd_output, _ = model_img(image)
            #-----印象側-----
            _, imp_feature, _ = model_imp(w2v)
            #----------concatenate----------
            if i==0:
                img_features = grd_output
                imp_features = imp_feature
            else:
                img_features =  torch.concatenate((img_features, grd_output), dim=0)
                imp_features =  torch.concatenate((imp_features, imp_feature), dim=0)
    img_features = img_features.to('cpu').detach().numpy().copy()
    imp_features = imp_features.to('cpu').detach().numpy().copy()
    return img_features, imp_features



def EmbedTag(model_imp, imp_list):
    """
    印象の中間表現を計算(タグ単体)
    """
    imp_vec_paths = '20240126/dataset/impression_vector/'
    with torch.no_grad():
        for i, imp in enumerate(imp_list):
            imp_path = imp_vec_paths + imp
            imp_vector = np.load(f"{imp_path}.npz")["arr_0"].astype(np.float32)
            word_vector = torch.from_numpy(imp_vector).type(torch.FloatTensor)
            _, imp_feature, _ = model_imp([word_vector])
            if i==0:
                imp_features = imp_feature
            else:
                imp_features =  torch.concatenate((imp_features, imp_feature), dim=0)
    imp_features = imp_features.to('cpu').detach().numpy().copy()
    return imp_features


def EmbedMultiTag(model_imp, query_tags):
    """
    印象の中間表現を計算(複数タグ)
    """
    #サイズ1のリストにタグ数×次元

    for i, tag in enumerate(query_tags):
        if i==0:
            w2v = torch.from_numpy(np.load(f"20240126/dataset/impression_vector/{tag}.npz")["arr_0"])
        else:
            w2v = torch.vstack((w2v, torch.from_numpy(np.load(f"20240126/dataset/impression_vector/{tag}.npz")["arr_0"])))
    with torch.no_grad():
        _, imp_feature, _ = model_imp([w2v])
    imp_feature = imp_feature.to('cpu').detach().numpy().copy()
    return imp_feature


def GetImpList():
    with open(f"20240126/dataset/tag_freq_top10.csv") as f:
        reader = csv.reader(f)
        rows = np.asarray([row for row in reader])[1:]
    tag_freq = np.asarray(rows[:,2], dtype=np.int32)
    imp_list = rows[:,0][tag_freq>=50]
    return imp_list




def RankMatrix(query, key):
    query = np.asarray(query, dtype=np.float64)
    key = np.asarray(key, dtype=np.float64)
    distance_matrix = np.zeros((len(query), len(key)))
    for i in range(len(query)):
        distance_matrix[i,:] = (np.linalg.norm(query[i]-key, axis=1)**2)/len(query[i])
    nearest_idx = np.argsort(distance_matrix, axis=1)
    rank_matrix = np.zeros_like(nearest_idx)
    rows, cols = np.indices(nearest_idx.shape)
    rank_matrix[rows, nearest_idx] = cols
    return rank_matrix, distance_matrix



def GetFontnames(dataset):
    """
    使用するフォントの名前を取得
    """
    img_dir = f"20240126/dataset/font_jihun/{dataset}"
    font_names = []
    for font_name in sorted(glob.glob(img_dir + "/*")):
        font_names.append(font_name[len(img_dir)+1:-4])
    return font_names


def GetFontTags(tag_path):
    """
    Args:
        タグのパス
    Returns:
        タグ
    """
    with open(tag_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        tags = [row for row in reader][0]
    return tags

def GetTagFreq(dataset):
    """
    datasetのタグの頻度を返す
    """
    with open(f"20240126/dataset/tag_freq_top10.csv") as f:
        reader = csv.reader(f)
        rows = np.asarray([row for row in reader])[1:]
    freq_dict = {}
    if dataset=="train": 
        l = 2
    elif dataset=="val":
        l = 3
    elif dataset=="test":
        l = 4
    tag_freq_train = np.asarray(rows[:,2], dtype=np.int32)
    tag_freq = np.asarray(rows[:,l], dtype=np.int32)[tag_freq_train>=50]
    for i in range(len(tag_freq)):
        freq_dict[rows[i,0]] = tag_freq[i]
    return freq_dict


def LoadDatasetPaths(dataset):
    """
    datasetのフォントとタグのパスのリストを返す
    """
    with open(f"20240126/dataset/tag_txt/fontname_{dataset}.csv") as f:
        reader = csv.reader(f)
        font_names = np.asarray([row for row in reader])
    font_paths = [f"20240126/dataset/font_numpy/{dataset}/{font_name[0]}.npz" for font_name in font_names]
    tag_paths = [f"20240126/dataset/tag_txt/{dataset}/{font_name[0]}.csv" for font_name in font_names]
    return font_paths, tag_paths



def CulcSetMetrics(query_tags, tags):
    """
    タグの集合を比較する
    入力:   query_tags(検索のクエリとしたタグの集合)
            tag(検索によりピックアップされたタグ)
    出力:   precision, recall, f1-score
    """
    true_positives = len(set(tags)&set(query_tags))
    rec_size = len(tags)
    act_size = len(query_tags)
    precision = true_positives/rec_size
    recall = true_positives/act_size
    f1_score = 2*(precision*recall)/(precision+recall) if (precision+recall)!=0 else 0.0
    return precision, recall, f1_score



def CalcAP_Tag2Img(img_features, tag_features, imp_list, tag_paths):
    """
    画像特徴ベクトルとタグ単体の印象特徴ベクトルを受け取って, タグ単体から画像を検索したときのAPを計算する 
    """
    _, distance_matrix = RankMatrix(tag_features, img_features)
    topk_args = np.argsort(distance_matrix, axis=1)
    AP = [0]*len(tag_features)
    for t in range(len(tag_features)):
        p = []
        count=0
        for k in range(len(img_features)):
            tags = GetFontTags(tag_paths[topk_args[t][k]])
            if imp_list[t] in tags:
                count += 1
                p.append(count/(k+1))
        AP[t]= np.sum(p)/count
    return AP




def CalcAP_Img2Tag(img_features, tag_features, imp_list, tag_paths):
    """
    画像特徴ベクトルとタグ単体の印象特徴ベクトルを受け取って, 画像からタグを検索したときのAPを計算する 
    """
    _, distance_matrix = RankMatrix(img_features, tag_features)
    topk_args = np.argsort(distance_matrix, axis=1)
    AP = [0]*len(img_features)
    number_of_tags = [0]*len(img_features)
    for f in range(len(img_features)):
        #----------フォントが持つタグの取得----------
        query_tags = GetFontTags(tag_paths[f])
        number_of_tags[f] = len(query_tags)
        p = []
        count=0
        for k in range(len(tag_features)):
            #-----近傍のタグがクエリに入っていればflag=1-----
            if imp_list[topk_args[f][k]] in query_tags:
                count += 1
                p.append(count/(k+1))
        AP[f]= np.sum(p)/count
    return AP, number_of_tags