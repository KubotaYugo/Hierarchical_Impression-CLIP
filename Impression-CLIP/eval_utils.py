"""
モデルの評価に用いる関数
"""
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import csv
from pathlib import Path
DIR_PAHT = Path(__file__).resolve().parent.parent.parent # /ICDAR_Kubotaまでのパスを取得





class DatasetForTag(Dataset):
    """
    タグだけのdataloderを作成
    入力:   タグのリスト
            tokenizer
    出力:   dataloder
    """
    def __init__(self, tag_list, tokenizer):
        self.tag_list = tag_list
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.tag_list)
    def __getitem__(self, idx):
        tag = self.tag_list[idx]
        prompt = f"The impression is {tag}."
        tokenized_text = self.tokenizer(prompt, truncate=True)
        return tokenized_text[0]
    


def EmbedFontText(dataloder, font_autoencoder, clip_model, emb_f, emb_t, device):
    """
    画像・印象語ペアを受け取ってencode&embed
    入力:   画像・印象語ペアのdataloder, 
            モデルの構成要素(font_autoencoder, clip_model, emb_f, emb_t)
            device
    出力:   埋め込み前の画像/印象特徴，埋め込み後の画像/印象特徴
    """
    with torch.no_grad():
        for i, data in enumerate(dataloder):
            #----------データの読み込み----------
            font = Variable(data[0]).to(device)
            tokenized_text = Variable(data[1]).to(device)
            #----------encode&embed----------
            font_feature = font_autoencoder.encoder(font)
            text_feature = clip_model.encode_text(tokenized_text)
            font_embedded_feature = emb_f(font_feature)
            text_embedded_feature = emb_t(text_feature.to(dtype=torch.float32))
            font_embedded_feature = ((font_embedded_feature.T/torch.norm(font_embedded_feature, dim=(1))).T)
            text_embedded_feature = ((text_embedded_feature.T/torch.norm(text_embedded_feature, dim=(1))).T)
            #----------concatenate----------
            if i==0:
                font_features = font_feature
                text_features = text_feature
                font_embedded_features = font_embedded_feature
                text_embedded_features = text_embedded_feature
            else:
                font_features =  torch.concatenate((font_features, font_feature), dim=0)
                text_features =  torch.concatenate((text_features, text_feature), dim=0)
                font_embedded_features = torch.concatenate((font_embedded_features, font_embedded_feature), dim=0)
                text_embedded_features = torch.concatenate((text_embedded_features, text_embedded_feature), dim=0)
    return font_features, text_features, font_embedded_features, text_embedded_features

def EmbedText(dataloder, clip_model, emb_t, device):
    """
    テキストを受け取ってencode&embed
    入力:   テキスト(tokenized)のdataloder
            印象側のモデルの構成要素(clip_model, emb_t)
            device
    出力:   埋め込み前の画像/印象特徴，埋め込み後の画像/印象特徴
    """
    with torch.no_grad():
        for i, data in enumerate(dataloder):
            #----------データの取得&eoncode&embed----------
            tokenized_text = Variable(data).to(device)
            text_feature = clip_model.encode_text(tokenized_text)
            text_embedded_feature = emb_t(text_feature.to(dtype=torch.float32))
            text_embedded_feature = ((text_embedded_feature.T/torch.norm(text_embedded_feature, dim=(1))).T)
            #----------concatenate----------
            if i==0:
                text_features = text_feature
                text_embedded_features = text_embedded_feature
            else:
                text_features =  torch.concatenate((text_features, text_feature), dim=0)
                text_embedded_features = torch.concatenate((text_embedded_features, text_embedded_feature), dim=0)
    return text_features, text_embedded_features



def CalcRetrievalRank(similarity_matrix, mode=None):
    """
    類似度行列を受け取って, 印象から画像or画像から印象を検索したときのRetrievalRankを計算する 
    入力:
        similarity_matrix: 類似度行列
        mode:
            mode=="Img2Imp": 画像から印象を検索
            mode=="Imp2Img": 印象から画像を検索
    """
    if mode=="Img2Imp":
        sorted_index = torch.argsort(similarity_matrix, dim=1, descending=True)
    elif mode=="Imp2Img":
        similarity_matrix = similarity_matrix.T
        sorted_index = torch.argsort(similarity_matrix, dim=1, descending=True)
    rank = [torch.where(sorted_index[i]==i)[0].item()+1 for i in range(sorted_index.shape[0])]
    return rank



def CalcAP_Tag2Img(font_embedded_features, tag_embedded_features, tag_list, tag_paths):
    """
    画像特徴ベクトルとタグ単体の印象特徴ベクトルを受け取って, タグ単体から画像を検索したときのAPを計算する 
    """
    similarity_matrix = torch.matmul(font_embedded_features, tag_embedded_features.T).to("cpu").detach().numpy()
    topk_args = np.argsort(-similarity_matrix, axis=0)  #降順にするため，-logitsに
    AP = [0]*len(tag_embedded_features)
    for t in range(len(tag_embedded_features)):
        p = []
        count=0
        for k in range(len(font_embedded_features)):
            tags = GetFontTags(tag_paths[topk_args[k][t]])
            if tag_list[t] in tags:
                count += 1
                p.append(count/(k+1))
        AP[t]= np.sum(p)/count
    return AP

def CalcAP_Img2Tag(font_embedded_features, tag_embedded_features, tag_list, tag_paths):
    """
    画像特徴ベクトルとタグ単体の印象特徴ベクトルを受け取って, 画像からタグ単体を検索したときのAPを計算する 
    """
    similarity_matrix = torch.matmul(font_embedded_features, tag_embedded_features.T).to("cpu").detach().numpy()
    topk_args = np.argsort(-similarity_matrix, axis=1)  #降順にするため，-logitsに
    AP = [0]*len(font_embedded_features)
    for f in range(len(font_embedded_features)):
        p = []
        query_tags = GetFontTags(tag_paths[f])
        count=0
        for k in range(len(tag_embedded_features)):
            if tag_list[topk_args[f][k]] in query_tags:
                count += 1
                p.append(count/(k+1))
        AP[f]= np.sum(p)/count
    return AP



def GetTagList():
    """
    評価対象とするタグをdictで返す
    """
    #-----実験に使用するタグの種類の取得(頻度順)-----
    with open(f"{DIR_PAHT}/dataset/preprocess/tag_freq_top10.csv") as f:
        reader = csv.reader(f)
        rows = np.asarray([row for row in reader])[1:]
    tag_freq = np.asarray(rows[:,2], dtype=np.int32)
    tag_list_org = rows[:,0][tag_freq>=50]
    #-----タグ番号→タグのdict作成-----
    tag_list = {}
    for i, tag in enumerate(tag_list_org):
        tag_list[i] = tag
    return tag_list


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

def CulculateRank(query, key):
    """
    summry:
        queryでkeyを検索したとき,の順位を要素に持つ
    input:
        query:  クエリとする特徴ベクトルの配列
        key:    キーとする特徴ベクトルの配列
    output:
        rank:   rank[i][j]はクエリ[i]でキーを検索したときのキー[j]の順位
    """
    query = np.asarray(query, dtype=np.float64)
    key = np.asarray(key, dtype=np.float64)
    distances = np.zeros((len(query), len(key)))
    for i in range(len(query)):
        distances[i,:] = (np.linalg.norm(query[i]-key, axis=1)**2)/len(query[i])
    nearest_idx = np.argsort(distances, axis=1)
    rank = np.zeros_like(nearest_idx)
    rows, cols = np.indices(nearest_idx.shape)
    rank[rows, nearest_idx] = cols
    return rank








def GetFontnames(dataset):
    """
    datasetのフォントの名前を返す
    """
    with open(f"20240126/dataset/tag_txt/fontname_{dataset}.csv") as f:
        reader = csv.reader(f)
        fontnames = np.asarray([row[0] for row in reader])
    return fontnames

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

"""
def Tag2Prompt(tags):
        prompt = ""
        if len(tags) >= 2:
            ordinal = ["First", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
            
            prompt1 = ordinal[0]
            prompt2 = tags[0]
            i = 0
            for i in range(1, min(len(tags)-1, 10-1)):
                prompt1 = prompt1 + ", " + ordinal[i]
                prompt2 = prompt2 + ", " + tags[i]
            prompt1 = prompt1 + " and " + ordinal[i+1] + " impressions are "
            prompt2 = prompt2 + " and " + tags[i+1] + ", respectively."                
            prompt = prompt1 + prompt2
        else:
            prompt = f"The impression is {tags[0]}."
        return prompt
"""
    
    
def ABCHERONS(font_path):
    pad_h = np.ones(shape=(64, 1))*255
    img = np.load(font_path)["arr_0"].astype(np.float32)
    images = img[0]
    for c in [1, 2, 7, 4, 17, 14, 13, 18]:
        images = np.hstack([images, pad_h, img[c]])
    return images