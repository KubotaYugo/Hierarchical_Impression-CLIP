'''
CLIPとSimCLR(階層構造なし)の対応付け精度(Retrieval Rank)を比較する
CLIP:   ImpressionCLIP/experiment2/stop_by_ARR
SimCLR: Hierarchical_ImpressionCLIP/experiment4/experiment4-1
'''

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)
from models import FontAutoencoder
from models import MLP
from lib import utils
from lib import eval_utils


def culc_retrieval_rank(embedded_img_features, embedded_tag_features):
    similarity_matrix = torch.matmul(embedded_img_features, embedded_tag_features.T)
    RR_tag2img = eval_utils.retrieval_rank(similarity_matrix, "tag2img")
    RR_img2tag = eval_utils.retrieval_rank(similarity_matrix, "img2tag")
    return RR_tag2img, RR_img2tag

def save_difference(RR_CLIP, RR_SimCLR, mode):
    # Retrieval Rankの差分をcsvに保存
    RR_CLIP = np.array(RR_CLIP)
    RR_SimCLR = np.array(RR_SimCLR)
    difference = RR_CLIP-RR_SimCLR
    fontnames = utils.get_fontnames(DATASET)
    filename = f'{SAVE_DIR}/{mode}.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['fontname', 'diffrence', 'without', 'with'])
        for i in range(len(RR_CLIP)):    
            writer.writerow([fontnames[i], difference[i], RR_CLIP[i], RR_SimCLR[i]])
    
    # SimCLRの方が良いもの(RRが小さいもの)top10とworst10の画像と印象タグを保存
    sort_indices = np.argsort(-1*difference)
    sorted_fontnames_RR_tag2img = fontnames[sort_indices]
    eval_utils.save_fonts(sorted_fontnames_RR_tag2img[:10], DATASET, f'{SAVE_DIR}/{mode}_top10')     # top10の保存
    eval_utils.save_fonts(sorted_fontnames_RR_tag2img[-10:], DATASET, f'{SAVE_DIR}/{mode}_worst10')  # worst10の保存


def plot_RR(RR_CLIP, RR_SimCLR, mode):
    # 横軸:RR_CLIP, 縦軸:RR_SimCLRとしてプロット
    def update_annot(ind):
        i = ind["ind"][0]
        pos = sc.get_offsets()[i]
        index = i%len(img_paths)
        fontname = img_paths[index][len(f"dataset/MyFonts_preprocessed/font_numpy_Impression-CLIP/{DATASET}/"):-4]
        annot_img.xy = (pos[0]+100, pos[1]+100)
        annot_text.xy = (pos[0]+30, pos[1]-80)
        img = np.load(img_paths[index])["arr_0"][0]
        imagebox.set_data(img)
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

    fig, ax = plt.subplots(figsize=(6.4*1.5, 4.8*1.5))
    plt.plot([1, len(RR_CLIP)], [1, len(RR_CLIP)], color='gray', zorder=1, linewidth=0.5)
    sc = plt.scatter(RR_CLIP, RR_SimCLR, s=3, zorder=2)

    img_paths, tag_paths = utils.LoadDatasetPaths(DATASET)
    img = np.load(img_paths[0])["arr_0"][0]
    imagebox = OffsetImage(img, zoom=0.7, cmap='gray')
    imagebox.image.axes = ax
    annot_img = AnnotationBbox(imagebox, xy=(0,0), xycoords="data", boxcoords="offset points", pad=0,
                            arrowprops=dict( arrowstyle="->", connectionstyle="arc3,rad=-0.3"))
    annot_img.set_visible(False)
    ax.add_artist(annot_img)
    annot_text = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points", bbox=dict(boxstyle="round", fc="w"))
    annot_text.set_visible(False)
    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.xlabel(f'RR_{mode}_CLIP')
    plt.ylabel(f'RR_{mode}_SimCLR')
    plt.savefig(f'{SAVE_DIR}/comp_RR_{mode}.png', bbox_inches='tight', dpi=500)
    plt.show()
    plt.close()



# define constant
DATASET = 'test'

# 保存用ディレクトリの作成
EXP = 'Hierarchical_Impression-CLIP/experiment3/experiment3-2'
LR = utils.LR
BATCH_SIZE = utils.BATCH_SIZE
SAVE_DIR = f'{EXP}/LR={LR}, BS={BATCH_SIZE}/comp_CLIP_SimCLR/comp_RR/{DATASET}'
os.makedirs(SAVE_DIR, exist_ok=True)


# CLIP: 特徴量の読み込み
load_dir = f'Impression-CLIP/experiment2/stop_by_ARR/features/{DATASET}'
embedded_img_features_CLIP = torch.load(f'{load_dir}/embedded_img_features.pth')
embedded_tag_features_CLIP = torch.load(f'{load_dir}/embedded_tag_features.pth')
# CLIP: Retrieval Rankの計算
RR_tag2img_CLIP, RR_img2tag_CLIP = culc_retrieval_rank(embedded_img_features_CLIP, embedded_tag_features_CLIP)

# SimCLR: 特徴量の読み込み
load_dir = f'Hierarchical_Impression-CLIP/experiment4/experiment4-1/LR={LR}, BS={BATCH_SIZE}/features/{DATASET}'
embedded_img_features_SimCLR = torch.load(f'{load_dir}/embedded_img_features.pth')
embedded_tag_features_SimCLR = torch.load(f'{load_dir}/embedded_tag_features.pth')
# SimCLR: Retrieval Rankの計算
RR_tag2img_SimCLR, RR_img2tag_SimCLR = culc_retrieval_rank(embedded_img_features_SimCLR, embedded_tag_features_SimCLR)


# Retrieval Rankの差分, 差分がtop10, worst10の画像/印象タグを保存
save_difference(RR_tag2img_CLIP, RR_tag2img_SimCLR, 'tag2img')
save_difference(RR_img2tag_CLIP, RR_img2tag_SimCLR, 'img2tag')

# プロットを保存
plot_RR(RR_tag2img_CLIP, RR_tag2img_SimCLR, 'tag2img')
plot_RR(RR_img2tag_CLIP, RR_img2tag_SimCLR, 'img2tag')