'''
画像特徴をbisecting kmeansでクラスタリング
'''

import torch
import os
import FontAutoencoder
import utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from openTSNE import TSNE
from sklearn.decomposition import PCA
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


MODEL_PATH = "FontAutoencoder/model/best.pt"
DATASET = "train"

# データの準備
font_paths, tag_paths = utils.LoadDatasetPaths(DATASET)
dataset = utils.ImageDataset(font_paths)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers = os.cpu_count(), pin_memory=True)

# モデルの準備
device = "cuda"
model = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# 画像特徴の取得
with torch.no_grad():
    for i, data in enumerate(dataloader):
        input_font = Variable(data).to(device)
        font_feature = model.encoder(input_font)
        if i==0:
            font_features = font_feature.to("cpu").detach().numpy()
        else:
            font_features = np.concatenate([font_features, font_feature.to("cpu").detach().numpy()])


# bisecting kmeans
NUM_CLUSTERS = 10
clusters, data_index = utils.bisecting_kmeans(font_features, NUM_CLUSTERS)

# 表現形式を変えて, data_indexを保存
data_index_trasform = np.zeros(len(font_features))
for i in range(len(data_index)):
    data_index_trasform[data_index[i]] = i
np.savez_compressed('image_clusters.npz', data_index_trasform)


# tSNE
# PERPLEXITY = 30
# N_ITER = 300
# embedding = TSNE(perplexity=PERPLEXITY, n_iter=N_ITER, initialization="pca", metric="euclidean", n_jobs=10, random_state=7).fit(font_features)
# X = embedding[:, 0]
# Y = embedding[:, 1]
# plt.figure(figsize=(8, 6))
# for i, indexes in enumerate(data_index):
#     indexes = np.asarray(indexes)
#     plt.scatter(X[indexes], Y[indexes], color=plt.cm.tab20(i), label=f'Cluster {i + 1}', s=3)
# plt.title('Bisecting K-means Clustering')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend()
# plt.savefig("image_clusters_tSNE.png")
# plt.close()

# PCA
pca = PCA(n_components=100)
pca.fit(font_features)
embedding = pca.transform(font_features)    
X = embedding[:, 0]
Y = embedding[:, 1]
plt.figure(figsize=(8, 6))
for i, indexes in enumerate(data_index):
    indexes = np.asarray(indexes)
    plt.scatter(X[indexes], Y[indexes], color=plt.cm.tab10(i), label=f'Cluster {i + 1}', s=3)
plt.title('Bisecting K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.savefig("image_clusters_PCA.png")
plt.close()


# マウスオーバーで画像とクラス，ファイル名を表示
fig, ax = plt.subplots()
img = np.load(font_paths[0])["arr_0"][0]
imagebox = OffsetImage(img, zoom=0.7, cmap='gray')
imagebox.image.axes = ax


sc = plt.scatter(X, Y, c=plt.cm.tab10(np.asarray(data_index_trasform, dtype=np.int64)), alpha=0.8, edgecolors='w')

annot_img = AnnotationBbox(imagebox, xy=(0,0), xycoords="data", boxcoords="offset points", pad=0,
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
    img = np.load(font_paths[i])["arr_0"][0]
    imagebox.set_data(img)
    annot_text.xy = (pos[0]+10, pos[1]-10)
    # text = f"{font_paths[i][len("dataset/MyFonts_preprocessed/"):]}"
    text = utils.get_tags(tag_paths[i])
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
plt.show()