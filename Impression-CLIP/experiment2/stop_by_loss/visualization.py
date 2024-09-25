import os
from transformers import CLIPTokenizer, CLIPModel
import FontAutoencoder
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import utils
import eval_utils
import numpy as np
import MLP

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from openTSNE import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns


# define constant
EXP = utils.EXP
MODEL_PATH = f"{EXP}/model/best.pth.tar"
BATCH_SIZE = utils.BATCH_SIZE
DATASET = 'test'
SAVE_PATH = f'{EXP}/visualization'
os.makedirs(SAVE_PATH, exist_ok=True)

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
dataset = utils.CustomDataset(img_paths, tag_paths, tokenizer)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# 単体タグのdataloder
tag_list = list(eval_utils.get_tag_list().values())
tagset = eval_utils.CustomDatasetForTag(tag_list, tokenizer)
tagloader = torch.utils.data.DataLoader(tagset, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# 特徴量の取得
_, _, embedded_img_features, embedded_tag_features = eval_utils.extract_features(font_autoencoder, clip_model, emb_i, emb_t, dataloader)
embedded_img_features = embedded_img_features.to('cpu').detach().numpy().copy()
embedded_tag_features = embedded_tag_features.to('cpu').detach().numpy().copy()
features = np.concatenate([embedded_img_features, embedded_tag_features], axis=0)

# PCA
N = 5
pca = PCA(n_components=100)
pca.fit(features)
embedding = pca.transform(features)
df = pd.DataFrame(embedding[:, :N])
df = df.assign(modal="img")
t = len(embedded_img_features)
df.loc[t:, "modal"] = "tag"

# 主成分方向の分布
sns.pairplot(df, hue="modal", plot_kws={'s':10})
plt.savefig(f"{SAVE_PATH}/PCA.png", bbox_inches='tight', dpi=500)
plt.close()

# 累積寄与率
plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.plot([0] + list(pca.explained_variance_ratio_), "-o")
plt.xlim(0, 100)
plt.ylim(0, 1.0)
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution rate")
plt.grid()
plt.savefig(f"{SAVE_PATH}/PCA_contribution.png", bbox_inches='tight', dpi=300)
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
plt.savefig(f"{SAVE_PATH}/PC1_PC2.png", bbox_inches='tight', dpi=500)
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

labels = [0]*len(embedded_img_features) + [1]*len(embedded_tag_features)
# labels = [0, 1]
# patches = [mpatches.Patch(color=plt.cm.tab10(i), label=f"img") for i in range(len(embedded_img_features))] + [mpatches.Patch(color=plt.cm.tab10(i), label=f"tag") for i in range(len(embedded_tag_features))]
patches = [mpatches.Patch(color=plt.cm.tab10(0), label=f"img"), mpatches.Patch(color=plt.cm.tab10(1), label=f"tag")]
sc = plt.scatter(X, Y, c=plt.cm.tab10(np.asarray(labels, dtype=np.int64)), alpha=0.8, edgecolors='w', linewidths=0.1, s=10)

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
plt.savefig(f"{SAVE_PATH}/tSNE.png", bbox_inches='tight', dpi=500)
plt.legend(handles=patches)
plt.show()