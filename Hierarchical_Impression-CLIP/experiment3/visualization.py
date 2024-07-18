"""
pretrain.pyで学習したモデルが出力する特徴量の分布を可視化する
"""

import os
from DMH import DMH_D_Eval
from transformers import CLIPTokenizer, CLIPModel
import FontAutoencoder
import MLP
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import losses
import numpy as np
import random
import csv
import utils
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from openTSNE import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns


# define constant
EXP = "experiment2"
SAVE_FOLDER = f"Hierarchical_Impression-CLIP/{EXP}/results/visualization"
MODEL_PATH = f"Hierarchical_Impression-CLIP/{EXP}/results/model/best.pth.tar"
IMG_HIERARCHY_PATH = "image_clusters.npz"
TAG_HIERARCHY_PATH = "impression_clusters.npz"
BATCH_SIZE = 256
DATASET = 'train'
ANNOTATE_WITH = 'tag'

# 保存用フォルダの準備
os.makedirs(f"{SAVE_FOLDER}", exist_ok=True)

# モデルの準備
device = torch.device('cuda:0')
font_autoencoder = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
emb_i = MLP.ReLU().to(device)
emb_t = MLP.ReLU().to(device)

# パラメータの読み込み
params = torch.load(MODEL_PATH)
font_autoencoder.load_state_dict(params['font_autoencoder'])
clip_model.load_state_dict(params['clip_model'])
emb_i.load_state_dict(params['emb_i'])
emb_t.load_state_dict(params['emb_t'])

# データの準備
img_paths, tag_paths = utils.LoadDatasetPaths(DATASET)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
dataset = DMH_D_Eval(img_paths, tag_paths, IMG_HIERARCHY_PATH, TAG_HIERARCHY_PATH, tokenizer)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, pin_memory=True)

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


# PCA
N = 5
pca = PCA(n_components=100)
pca.fit(features)
embedding = pca.transform(features)
df = pd.DataFrame(embedding[:, :N])
df = df.assign(modal="img")
t = len(img_features_stack)
df.loc[t:, "modal"] = "tag"

# 主成分方向の分布
sns.pairplot(df, hue="modal", plot_kws={'s':10})
plt.savefig(f"{SAVE_FOLDER}/PCA.png", bbox_inches='tight', dpi=500)
plt.close()

# 累積寄与率
plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.plot([0] + list(pca.explained_variance_ratio_), "-o")
plt.xlim(0, 100)
plt.ylim(0, 1.0)
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution rate")
plt.grid()
plt.savefig(f"{SAVE_FOLDER}/PCA_contribution.png", bbox_inches='tight', dpi=300)
plt.close()

# 第1，第2主成分方向のプロット
X = embedding[:,0]
Y = embedding[:,1]
fig, ax = plt.subplots(figsize=(16, 12))
plt.scatter(X[:t], Y[:t], c='#377eb8', label="img", alpha=0.8, s=5)
plt.scatter(X[t:], Y[t:], c='#ff7f00', label="tag", alpha=0.8, s=5)
plt.legend()
plt.xlim(X.min(), X.max())
plt.ylim(Y.min(), Y.max())
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig(f"{SAVE_FOLDER}/PC1_PC2.png", bbox_inches='tight', dpi=500)
plt.close()


# tSNE
PERPLEXITY = 30
N_ITER = 300
print("tSNE_start")
embedding = TSNE(perplexity=PERPLEXITY, n_iter=N_ITER, initialization="pca", metric="euclidean", n_jobs=10, random_state=7).fit(features)
print("tSNE_end")
X = embedding[:, 0]
Y = embedding[:, 1]

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

annot_img = AnnotationBbox(imagebox, xy=(0,0), xycoords="data", boxcoords="offset points", pad=0,
                           arrowprops=dict( arrowstyle="->", connectionstyle="arc3,rad=-0.3"))
annot_img.set_visible(False)
ax.add_artist(annot_img)

annot_text = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points", bbox=dict(boxstyle="round", fc="w"))
annot_text.set_visible(False)

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

fig.canvas.mpl_connect("motion_notify_event", hover)
plt.savefig(f"{SAVE_FOLDER}/tSNE_{ANNOTATE_WITH}.png", bbox_inches='tight', dpi=500)
plt.legend(handles=patches)
plt.show()