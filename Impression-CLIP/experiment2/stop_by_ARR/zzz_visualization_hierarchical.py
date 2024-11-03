import os
from transformers import CLIPTokenizer, CLIPModel
import FontAutoencoder
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data.dataset import Dataset
import utils
import eval_utils
import numpy as np
import MLP
import csv

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from openTSNE import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns


class DMH_D_Eval_With_Labels(Dataset):
    def __init__(self, img_paths, tag_paths, img_hierarchy_path, tag_hierarchy_path, tokenizer):
        self.tokenizer = tokenizer
        self.img_paths = img_paths
        self.tag_paths = tag_paths
        self.img_hierarchy = np.load(img_hierarchy_path)["arr_0"].astype(np.int64)
        self.tag_hierarchy = np.load(tag_hierarchy_path)["arr_0"].astype(np.int64)
        self.num_data = len(img_paths)
        self.img_labels = {}
        self.tag_labels = {}
        for i in range(self.num_data):
            img_cluster = self.img_hierarchy[i]
            tag_cluster = self.tag_hierarchy[i]
            # 画像のモダリティの階層構造
            if img_cluster not in self.img_labels:
                self.img_labels[img_cluster] = {}
            self.img_labels[img_cluster][i] = i       
            # 印象のモダリティの階層構造
            if tag_cluster not in self.tag_labels:
                self.tag_labels[tag_cluster] = {}
            self.tag_labels[tag_cluster][i] = i

    def get_token(self, tag_path):
        with open(tag_path, encoding='utf8') as f:
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
        return tokenized_text
        
    def __getitem__(self, index):
        imgs, tokenized_tags, img_labels, tag_labels = [], [], [], []
        img = np.load(self.img_paths[index])["arr_0"].astype(np.float32)
        img = torch.from_numpy(img/255)
        tokenized_tag = self.get_token(self.tag_paths[index])['input_ids'][0]
        img_label = [self.img_hierarchy[index], index]
        tag_label = [self.tag_hierarchy[index], index]
        imgs.append(img)
        tokenized_tags.append(tokenized_tag)
        img_labels.append(img_label)
        tag_labels.append(tag_label)
        return imgs, tokenized_tags, torch.tensor(img_labels), torch.tensor(tag_labels)

    def __len__(self):
        return self.num_data
    

def update_annot(ind):
    i = ind["ind"][0]
    pos = sc.get_offsets()[i]
    index = i%len(img_paths)
    fontname = img_paths[index][len(f"dataset/MyFonts_preprocessed/font_numpy_Impression-CLIP/{DATASET}/"):-4]
    if i<len(img_paths):
        annot_img.xy = (pos[0]+0.5, pos[1]+0.5)
        annot_text.xy = (pos[0]+0.3, pos[1]-0.3)
        img = np.load(img_paths[index])["arr_0"][0]
        imagebox.set_data(img)
        annot_text.set_text(fontname)
    else:
        annot_text.xy = (pos[0]+0.3, pos[1]-0.3)
        tags = utils.get_tags(tag_paths[index])
        annot_text.set_text(f"{fontname} {tags}")

def hover(event):
    vis = annot_img.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            if ind["ind"][0] < len(img_paths):
                annot_img.set_visible(True)
                annot_text.set_visible(True)
            else:
                annot_text.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot_img.set_visible(False)
                annot_text.set_visible(False)
                fig.canvas.draw_idle()



# define constant
EXP = utils.EXP
IMG_HIERARCHY_PATH = 'image_clusters.npz'
TAG_HIERARCHY_PATH = 'impression_clusters.npz'
MODEL_PATH = f"{EXP}/model/best.pth.tar"
BATCH_SIZE = utils.BATCH_SIZE
DATASET = 'train'
SAVE_DIR = f'{EXP}/visualization'
os.makedirs(SAVE_DIR, exist_ok=True)

# モデルの準備
device = torch.device('cuda:0')
font_autoencoder = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device)
font_encoder = font_autoencoder.encoder    
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
emb_i = MLP.ReLU().to(device)
emb_t = MLP.ReLU().to(device)

# パラメータの読み込み
params = torch.load(MODEL_PATH)
font_encoder.load_state_dict(params['font_encoder'])
clip_model.load_state_dict(params['clip_model'])
emb_i.load_state_dict(params['emb_i'])
emb_t.load_state_dict(params['emb_t'])
font_autoencoder.eval()
clip_model.eval()
emb_i.eval()
emb_t.eval()

# dataloderの準備
img_paths, tag_paths = utils.LoadDatasetPaths(DATASET)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
dataset = DMH_D_Eval_With_Labels(img_paths, tag_paths, IMG_HIERARCHY_PATH, TAG_HIERARCHY_PATH, tokenizer)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# 特徴量の取得
for idx, data in enumerate(dataloader):
    imgs, tokenized_tags, img_labels, tag_labels = data
    imgs, tokenized_tags, img_labels, tag_labels = imgs[0], tokenized_tags[0], img_labels[:,0], tag_labels[:,0]
    imgs = imgs.cuda(non_blocking=True)
    tokenized_tags = tokenized_tags.cuda(non_blocking=True)
    img_labels = img_labels.cuda(non_blocking=True)
    tag_labels = tag_labels.cuda(non_blocking=True)
    with torch.no_grad():
        img_features = font_autoencoder.encoder(imgs)
        tag_features = clip_model.get_text_features(tokenized_tags) 
        embedded_img_features = emb_i(img_features)
        embedded_tag_features = emb_t(tag_features)
    if idx==0:
        img_labels_stack = img_labels
        tag_labels_stack = tag_labels
        img_features_stack = embedded_img_features
        tag_features_stack = embedded_tag_features
    else:
        img_labels_stack = torch.concatenate((img_labels_stack, img_labels), dim=0)
        tag_labels_stack = torch.concatenate((tag_labels_stack, tag_labels), dim=0)
        img_features_stack = torch.concatenate((img_features_stack, embedded_img_features), dim=0)
        tag_features_stack = torch.concatenate((tag_features_stack, embedded_tag_features), dim=0)
img_labels_stack = img_labels_stack.to('cpu').detach().numpy().copy()[:,0]
tag_labels_stack = tag_labels_stack.to('cpu').detach().numpy().copy()[:,0]
img_features_stack = img_features_stack.to('cpu').detach().numpy().copy()
tag_features_stack = tag_features_stack.to('cpu').detach().numpy().copy()
features = np.concatenate([img_features_stack, tag_features_stack], axis=0)


# tSNE
PERPLEXITY = 30
N_ITER = 300
print("tSNE_start")
embedding = TSNE(perplexity=PERPLEXITY, n_iter=N_ITER, initialization="pca", metric="euclidean", n_jobs=10, random_state=7).fit(features)
print("tSNE_end")
X = embedding[:, 0]
Y = embedding[:, 1]

# 画像 or 印象のモダリティで色分け
fig, ax = plt.subplots(figsize=(6.4*1.5, 4.8*1.5))
img = np.load(img_paths[0])["arr_0"][0]

labels = [0]*len(img_labels_stack) + [1]*len(tag_labels_stack)
modality = ['img', 'tag']
patches = [mpatches.Patch(color=plt.cm.tab10(i), label=modality[i]) for i in range(2)]
sc = plt.scatter(X, Y, c=plt.cm.tab10(np.asarray(labels, dtype=np.int64)), alpha=0.8, edgecolors='w',
                linewidths=0.1, s=10)
plt.xlim(-90, 90)
plt.ylim(-90, 90)
# plt.legend(handles=patches)
plt.savefig(f"{SAVE_DIR}/tSNE_with_hierarchy_{DATASET}.png", bbox_inches='tight', dpi=500)
plt.show()
plt.close()


# 各モダリティのクラスタ別に色分け
for ANNOTATE_WITH in ['img', 'tag']:
    # マウスオーバーで画像とクラス，ファイル名を表示
    fig, ax = plt.subplots(figsize=(6.4*1.5, 4.8*1.5))
    img = np.load(img_paths[0])["arr_0"][0]
    imagebox = OffsetImage(img, zoom=0.7, cmap='gray')
    imagebox.image.axes = ax
    
    if ANNOTATE_WITH=='img':
        labels = list(img_labels_stack*2)+list(img_labels_stack*2+1)
    elif ANNOTATE_WITH=='tag':
        labels = list(tag_labels_stack*2)+list(tag_labels_stack*2+1)
    modality = ['img', 'tag']
    patches = [mpatches.Patch(color=plt.cm.tab20(i), label=f"cluster{i//2}_{modality[i%2]}") for i in range(20)]
    sc = plt.scatter(X, Y, c=plt.cm.tab20(np.asarray(labels, dtype=np.int64)), alpha=0.8, edgecolors='w',
                    linewidths=0.1, s=10)
    plt.xlim(-90, 90)
    plt.ylim(-90, 90)

    annot_img = AnnotationBbox(imagebox, xy=(0,0), xycoords="data", boxcoords="offset points", pad=0,
                            arrowprops=dict( arrowstyle="->", connectionstyle="arc3,rad=-0.3"))
    annot_img.set_visible(False)
    ax.add_artist(annot_img)

    annot_text = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points", bbox=dict(boxstyle="round", fc="w"))
    annot_text.set_visible(False)

    fig.canvas.mpl_connect("motion_notify_event", hover)
    # plt.legend(handles=patches)
    plt.savefig(f"{SAVE_DIR}/tSNE_with_hierarchy_{ANNOTATE_WITH}_{DATASET}.png", bbox_inches='tight', dpi=500)
    plt.show()
    plt.close()