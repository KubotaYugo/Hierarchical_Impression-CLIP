"""
pretrain.pyで学習したモデルが出力する特徴量の分布を可視化する
"""

import os
import json
import numpy as np
import torch
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from PIL import Image
import resnet_modified
import seaborn as sns
import matplotlib.pyplot as plt
from openTSNE import TSNE
from sklearn.decomposition import PCA


def set_model(model_path):
    model = resnet_modified.MyResNet(name='resnet50')
    state_dict = torch.load(model_path, map_location='cpu')["state_dict"]
    model.load_state_dict(state_dict)
    return model


def txt_parse(f):
    result = []
    with open(f) as fp:
        line = fp.readline()
        result.append(line)
        while line:
            line = fp.readline()
            result.append(line)
    return result


class DeepFashionHierarchihcalDataset(Dataset):
    def __init__(self, list_file, class_map_file, repeating_product_file, transform):
        self.transform = transform
        
        with open(list_file, 'r') as f:
            data_dict = json.load(f)

        with open(class_map_file, 'r') as f:
            self.class_map = json.load(f)
        self.repeating_product_ids = txt_parse(repeating_product_file)

        self.filenames = []
        self.category = []
        self.labels = {}
        # for i in range(len(data_dict['images'])):
        for i in range(10):
            filename = data_dict['images'][i]
            category = self.class_map[data_dict['categories'][i]]
            product, variation, image = self.get_label_split(filename)
            if product not in self.repeating_product_ids:
                if category not in self.labels:
                    self.labels[category] = {}
                if product not in self.labels[category]:
                    self.labels[category][product] = {}
                if variation not in self.labels[category][product]:
                    self.labels[category][product][variation] = {}
                self.labels[category][product][variation][image] = i
                self.category.append(category)
                self.filenames.append(filename)

    def get_label_split(self, filename):
        split = filename.split('/')
        image_split = split[-1].split('.')[0].split('_')
        return int(split[-2][3:]), int(image_split[0]), int(image_split[1])

    def get_label_split_by_index(self, index):
        filename = self.filenames[index]
        category = self.category[index]
        product, variation, image = self.get_label_split(filename)
        return [category, product, variation, image]

    def __getitem__(self, index):
        image = Image.open(self.filenames[index])
        image = self.transform(image)
        label = list(self.get_label_split_by_index(index))
        return image, torch.tensor(label)

    def __len__(self):
        return len(self.filenames)


# ハイパラ
MODEL_PATH = "DeepFashion/results_org/model/checkpoint_0100.pth.tar"
LIST_FILE = "dataset/listfile/test.json"
CLASS_MAP_FILE = "dataset/class_map.json"
REPEATING_PRODUCT_FILE = "dataset/repeating_product_ids.csv"
BATCH_SIZE = 512

# 保存用フォルダの準備
SAVE_PATH = f"results_org/visualization"
os.makedirs(f"{SAVE_PATH}", exist_ok=True)

# モデルの準備
device = torch.device('cuda:0')
model = set_model(MODEL_PATH).to(device)
model.eval()

# データの準備
transform = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                     [0.229, 0.224, 0.225])])
dataset = DeepFashionHierarchihcalDataset(LIST_FILE, CLASS_MAP_FILE, REPEATING_PRODUCT_FILE, transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = os.cpu_count(), pin_memory=True)

# 特徴量の取得
with torch.no_grad():
    for i, data in enumerate(dataloader):
        image = data[0].to(device)
        label = data[1]
        feature = model(image)
        if i==0:
            labels = label
            features = feature
        else:
            labels = torch.concatenate((labels, label), dim=0)
            features = torch.concatenate((features, feature), dim=0)
labels = labels.to('cpu').detach().numpy().copy()
features = features.to('cpu').detach().numpy().copy()


# # PCA
# N = 5
# pca = PCA(n_components=100)
# pca.fit(features)
# embedding = pca.transform(features)
# df = pd.DataFrame(embedding[:, :N])

# # 主成分方向の分布
# sns.pairplot(df, plot_kws={'s':10})
# plt.savefig(f"{SAVE_PATH}/PCA.png", dpi=500)
# plt.close()

# # 累積寄与率
# plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o")
# plt.plot([0] + list(pca.explained_variance_ratio_), "-o")
# plt.xlim(0, 100)
# plt.ylim(0, 1.0)
# plt.xlabel("Number of principal components")
# plt.ylabel("Cumulative contribution rate")
# plt.grid()
# plt.savefig(f"{SAVE_PATH}/PCA_contribution.png", dpi=300)
# plt.close()

# # 第1，第2主成分方向のプロット
# X = embedding[:,0]
# Y = embedding[:,1]
# fig, ax = plt.subplots(figsize=(16, 12))
# plt.scatter(X, Y, c='#377eb8')
# plt.legend()
# plt.xlim(X.min(), X.max())
# plt.ylim(Y.min(), Y.max())
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.savefig(f"{SAVE_PATH}/PC1_PC2.png", dpi=500)
# plt.close()


# tSNE
PERPLEXITY = 30
N_ITER = 300
print("tSNE_start")
embedding = TSNE(perplexity=PERPLEXITY, n_iter=N_ITER, initialization="pca", metric="euclidean", n_jobs=10, random_state=7).fit(features)
print("tSNE_end")
np.savez(f"{SAVE_PATH}/tSNE_numpy.npz", data=embedding)
embedding = np.load(f"{SAVE_PATH}/tSNE_numpy.npz")["data"]
X = embedding[:, 0]
Y = embedding[:, 1]

# # 最上位のクラスで色分けしてプロット
# with open(CLASS_MAP_FILE, 'r') as f:
#     class_map = json.load(f)
# class_map_reverse = {value: key for key, value in class_map.items()}

# plt.figure(figsize=(16, 12))
# NUM_CLASSES = 17
# for class_id in range(NUM_CLASSES):
#     plt.scatter(X[labels[:,0] == class_id], Y[labels[:,0] == class_id], 
#                 color=plt.cm.tab20(class_id), label=class_map_reverse[class_id], alpha=0.8, edgecolors='w')
# plt.legend()
# plt.savefig(f"{SAVE_PATH}/tSNE_class.png", dpi=300)
# plt.show()
# plt.close()


# # 画像でプロット
# with open(LIST_FILE, 'r') as f:
#     filenames = json.load(f)['images']
# fig, ax = plt.subplots(figsize=(16, 12))
# for i in range(len(features)):
#     image = np.asarray(Image.open(filenames[i]))
#     imagebox = OffsetImage(image, zoom=0.1)
#     ab = AnnotationBbox(imagebox, (X[i], Y[i]), frameon=True, pad=0)
#     ax.add_artist(ab)
# plt.xlim(X.min(), X.max())
# plt.ylim(Y.min(), Y.max())
# plt.savefig(f"{SAVE_PATH}/tSNE_image.png", dpi=300)
# plt.show()
# plt.close()


# マウスオーバーで画像とクラス，ファイル名を表示
fig, ax = plt.subplots()

with open(LIST_FILE, 'r') as f:
    filenames = json.load(f)['images']
img = np.asarray(Image.open(filenames[0]))
imagebox = OffsetImage(img, zoom=0.7)
imagebox.image.axes = ax

with open(CLASS_MAP_FILE, 'r') as f:
    class_map = json.load(f)
class_map_reverse = {value: key for key, value in class_map.items()}
NUM_CLASSES = 17
patches = [mpatches.Patch(color=plt.cm.tab20(i), label=class_map_reverse[i]) for i in range(NUM_CLASSES)]

sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=plt.cm.tab20(labels[:,0]), alpha=0.8, edgecolors='w')

annot_img = AnnotationBbox(imagebox, xy=(0,0),
                           xycoords="data", boxcoords="offset points", pad=0,
                           arrowprops=dict( arrowstyle="->", connectionstyle="arc3,rad=-0.3"))
annot_img.set_visible(False)
ax.add_artist(annot_img)

annot_text = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"))
annot_text.set_visible(False)

def update_annot(ind):
    i = ind["ind"][0]
    pos = sc.get_offsets()[i]
    annot_img.xy = (pos[0]+10, pos[1]+10)
    img = np.asarray(Image.open(filenames[i]))
    imagebox.set_data(img)
    annot_text.xy = (pos[0]+10, pos[1]-10)
    text = f"{labels[i]}\n{filenames[i][len("dataset/img/"):]}"
    annot_text.set_text(text)

def hover(event):
    vis = annot_img.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot_img.set_visible(True)
            annot_text.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot_img.set_visible(False)
                annot_text.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)
plt.legend(handles=patches)
plt.show()