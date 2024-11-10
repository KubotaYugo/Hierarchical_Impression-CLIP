import torch
from torch.utils.data import Dataset
import numpy as np
import csv
import random
import numpy as np
from sklearn.cluster import KMeans


# define constant
MAX_EPOCH = 10000
EARLY_STOPPING_PATIENCE = 100
FONT_AUTOENCODER_PATH = 'FontAutoencoder/model/best.pt'

EXP = 'Hierarchical_Impression-CLIP/experiment4/experiment4-2'
IMG_CLUSTER_PATH = 'Hierarchical_Impression-CLIP/experiment3/experiment3-4/clustering/train/image_clusters.npz'
TAG_CLUSTER_PATH = 'Hierarchical_Impression-CLIP/experiment3/experiment3-4/clustering/train/impression_clusters.npz'


# optimized parameters
LR = 1e-4
BATCH_SIZE = 32
BASE_DIR = f'{EXP}/LR={LR}_BS={BATCH_SIZE}'
MODEL_PATH = f'Hierarchical_Impression-CLIP/experiment4/experiment4-1/LR={LR}, BS={BATCH_SIZE}/results/model/best.pth.tar'


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

class EvalDataset(Dataset):
    """
    フォントとタグのdataloderを作成
    入力:   font_paths: フォントのパス
            tag_paths: タグのパス
            tokenizer
    出力:   dataloder
    """
    def __init__(self, font_paths, tag_paths, tokenizer):
        self.font_paths = font_paths
        self.tag_paths = tag_paths
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.font_paths)
    def __getitem__(self, idx):
        # フォント
        font = np.load(self.font_paths[idx])["arr_0"].astype(np.float32)
        font = torch.from_numpy(font/255)
        # タグ
        with open(self.tag_paths[idx], encoding='utf8') as f:
            csvreader = csv.reader(f)
            tags = [row for row in csvreader][0]
        if len(tags) == 1:
            prompt = f"The impression is {tags[0]}."
        elif len(tags) == 2:
            prompt = f"First and second impressions are {tags[0]} and {tags[1]}, respectively."
        elif len(tags) >= 3:
            ordinal = ["First", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
            prompt1 = ordinal[0]
            prompt2 = tags[0]
            i = 0
            for i in range(1, min(len(tags)-1, 10-1)):
                prompt1 = prompt1 + ", " + ordinal[i]
                prompt2 = prompt2 + ", " + tags[i]
            prompt1 = prompt1 + ", and " + ordinal[i+1] + " impressions are "
            prompt2 = prompt2 + ", and " + tags[i+1] + ", respectively."                
            prompt = prompt1 + prompt2
        tokenized_text = self.tokenizer(prompt, return_tensors="pt", max_length=self.tokenizer.max_model_input_sizes['openai/clip-vit-base-patch32'], padding="max_length", truncation=True)
        return font, tokenized_text['input_ids'][0]


class ImpressionDataset(Dataset):
    def __init__(self, tag_paths, tokenizer):
        self.tag_paths = tag_paths
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.tag_paths)
    def __getitem__(self, idx):
        with open(self.tag_paths[idx], encoding='utf8') as f:
            csvreader = csv.reader(f)
            tags = [row for row in csvreader][0]
        if len(tags) == 1:
            prompt = f"The impression is {tags[0]}."
        elif len(tags) == 2:
            prompt = f"First and second impressions are {tags[0]} and {tags[1]}, respectively."
        elif len(tags) >= 3:
            ordinal = ["First", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
            prompt1 = ordinal[0]
            prompt2 = tags[0]
            i = 0
            for i in range(1, min(len(tags)-1, 10-1)):
                prompt1 = prompt1 + ", " + ordinal[i]
                prompt2 = prompt2 + ", " + tags[i]
            prompt1 = prompt1 + ", and " + ordinal[i+1] + " impressions are "
            prompt2 = prompt2 + ", and " + tags[i+1] + ", respectively."                
            prompt = prompt1 + prompt2
        tokenized_text = self.tokenizer(prompt, return_tensors="pt", max_length=self.tokenizer.max_model_input_sizes['openai/clip-vit-base-patch32'], padding="max_length", truncation=True)
        return tokenized_text['input_ids'][0]

class ImageDataset(Dataset):
    def __init__(self, font_paths):
        self.font_paths = font_paths
    def __len__(self):
        return len(self.font_paths)
    def __getitem__(self, idx):
        font = np.load(self.font_paths[idx])["arr_0"].astype(np.float32)
        font = torch.from_numpy(font/255)
        return font


class EarlyStopping:
    def __init__(self, patience, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.early_stop = False
        self.min_val_loss = np.Inf
    def __call__(self, val_loss):
        if self.min_val_loss+self.delta <= val_loss:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            if  val_loss < self.min_val_loss:
                print(f'Validation loss decreased ({self.min_val_loss} --> {val_loss})')
                self.min_val_loss = val_loss


def calculate_within_cluster_variance(cluster):
    kmeans = KMeans(n_clusters=1).fit(cluster)
    center = kmeans.cluster_centers_
    variance = np.sum((cluster - center) ** 2)
    return variance

def bisecting_kmeans(data, num_clusters):
    clusters = [data]
    variances = [calculate_within_cluster_variance(data)]
    data_index = [[i for i in range(len(data))]]
    
    while len(clusters) < num_clusters:
        largest_variance_idx = np.argmax(variances)
        largest_cluster = clusters.pop(largest_variance_idx)
        largest_variance = variances.pop(largest_variance_idx)
        largest_index = data_index.pop(largest_variance_idx)
        
        kmeans = KMeans(n_clusters=2, random_state=42).fit(largest_cluster)
        labels = kmeans.labels_
        
        # 特徴量をクラスタを分割
        cluster_1 = largest_cluster[labels == 0]
        cluster_2 = largest_cluster[labels == 1]
        clusters.append(cluster_1)
        clusters.append(cluster_2)
        
        # クラスタを分割
        data_index_1  = np.asarray(largest_index)[labels == 0]
        data_index_2  = np.asarray(largest_index)[labels == 1]
        data_index.append(data_index_1)
        data_index.append(data_index_2)
        
        variances.append(calculate_within_cluster_variance(cluster_1))
        variances.append(calculate_within_cluster_variance(cluster_2))
        
    return clusters, data_index