'''
リスト内のフォントのタグと画像を保存
'''
import csv
import numpy as np
from matplotlib import cm
from matplotlib.image import imread, imsave

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils


DATASET = 'train'
FONTLIST = ['nagae']
FILENAME = f'compRR_img2tag_difference_worst10_{DATASET}'
EXP = utils.EXP
LR = utils.LR
BATCH_SIZE = utils.BATCH_SIZE
SAVE_DIR = SAVE_DIR = f'{EXP}/LR={LR}, BS={BATCH_SIZE}/savefonts'

os.makedirs(SAVE_DIR, exist_ok=True)

# 画像を保存
imgs_hstacks = []
pad_h = np.ones(shape=(64, 1))*255
pad_v = np.ones(shape=(3, 64*26+1*25))*255
for fontname in FONTLIST:
    fontpath = f'dataset/MyFonts_preprocessed/font_numpy_Impression-CLIP/{DATASET}/{fontname}.npz' 
    imgs = np.load(fontpath)["arr_0"].astype(np.float32)
    imgs_hstacks.append(np.hstack([imgs[0]]+[np.hstack([pad_h, imgs[i]]) for i in range(1, 26)]))
output_img = np.vstack([imgs_hstacks[0]]+[np.vstack([pad_v, imgs_hstacks[i]]) for i in range(1, len(imgs_hstacks))])
imsave(f'{SAVE_DIR}/{FILENAME}.png', output_img, cmap=cm.gray)

# タグを保存
write_rows = [['fontname', 'tags']]
for fontname in FONTLIST:
    tag_path = f'dataset/MyFonts_preprocessed/tag_txt/{DATASET}/{fontname}.csv'
    tags = utils.get_font_tags(tag_path)
    write_rows.append([fontname]+tags)
with open(f'{SAVE_DIR}/{FILENAME}.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(write_rows)