'''
myfontsデータセットのフォントの前処理
全フォントを, フォント中で一貫した倍率で64*64にリサイズして, numpyとpngで保存
処理の対象: font_flagが0のフォントは対象外 
           また，各フォントの大文字のみ
(1)フォント中の全文字で最も大きい縦幅or横幅を取得
(2)(1)をもとにした倍率でリサイズ
(3)リサイズした画像をnumpyとpngで保存
(4)抜けている文字があるフォントは除外しているはずなので，もしあったら強制終了
'''
from matplotlib import cm
from matplotlib.image import imsave
from PIL import Image
import numpy as np
import os
from decimal import Decimal, ROUND_HALF_UP
from multiprocessing import pool
import string

import utils_data_preprocessing as utils


def round_to_first_decimal(value):
    rounded_value = Decimal(str(value)).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    return int(rounded_value)

def save_fonts(values):
    # separate the variables
    idx = values[0]
    font_name = values[1]
    font_flag = values[2]
    # print(DATASET, idx)

    # exit without doing anything if the flag is zero
    if font_flag=='0':
        return None

    # load images of all characters in the font (target only uppercase letters)
    chars = string.ascii_uppercase 
    double_chars = [ch * 2 for ch in chars]
    char_paths = [f'{MYFONTS_DIR}/{font_name}_{double_char}.png' for double_char in double_chars]
    img_list = [np.array(Image.open(char_path).convert('L')) for char_path in char_paths]

    #　calculate the largest vertical or horizontal width among all characters
    max_length = 0
    for img in img_list:
        y_min = np.where(img!=255)[0].min()
        y_max = np.where(img!=255)[0].max()
        x_min = np.where(img!=255)[1].min()
        x_max = np.where(img!=255)[1].max()
        width = x_max-x_min if x_min!=x_max else 1
        height = y_max-y_min  if y_min!=y_max else 1
        max_length_char = np.max([width, height])
        if max_length_char > max_length:
            max_length = max_length_char
    char_scale = IMAGE_EDGE_LENGTH/max_length
    
    # comment
    font_list = []
    for img in img_list:
        # crop the character area
        y_min = np.where(img!=255)[0].min()
        y_max = np.where(img!=255)[0].max()
        x_min = np.where(img!=255)[1].min()
        x_max = np.where(img!=255)[1].max()
        image_clip = img[y_min:y_max+1, x_min:x_max+1]
        
        # resize
        scaled_width = round_to_first_decimal(char_scale*(x_max-x_min))
        scaled_height = round_to_first_decimal(char_scale*(y_max-y_min))
        scaled_width = 1 if scaled_width==0 else scaled_width
        scaled_height = 1 if scaled_height==0 else scaled_height
        image_clip = Image.fromarray(image_clip)
        img_resized = image_clip.resize([scaled_width, scaled_height], resample=Image.BICUBIC)
        img_resized = np.array(img_resized)

        # place the character at the center of the base image
        base_img = np.full([IMAGE_EDGE_LENGTH, IMAGE_EDGE_LENGTH], 255)
        y = img_resized.shape[0]
        x = img_resized.shape[1]
        y_top = round_to_first_decimal((IMAGE_EDGE_LENGTH-img_resized.shape[0])/2)
        x_left = round_to_first_decimal((IMAGE_EDGE_LENGTH-img_resized.shape[1])/2)
        base_img[y_top:y_top+y, x_left:x_left+x] = img_resized

        # normalize pixel values to the range 0-255
        base_img = base_img-np.min(base_img)
        base_img = base_img/np.max(base_img)*255
        font_list.append(base_img)
    stacked_font = np.asarray(font_list)

    # save preprocessed font as npz format
    np.savez_compressed(f'{SAVE_DIR_NPZ}/{font_name}.npz', stacked_font)

    # save preprocessed font as png format
    pad_h = np.ones(shape=(64, 1))*255
    font_image = [stacked_font[0]]
    for font in stacked_font[1:]:
        font_image.append(pad_h)
        font_image.append(font)
    font_image_stacked = np.hstack(font_image)
    imsave(f'{SAVE_DIR_IMG}/{font_name}.png', font_image_stacked, cmap=cm.gray)


if __name__ == '__main__':

    # difine parameters
    args = utils.get_args()
    DATASET = args.dataset
    IMAGE_EDGE_LENGTH = 64     # height and width of the image to be saved
    MYFONTS_DIR = 'MyFonts/fontimage'
    SAVE_DIR_IMG = f'MyFonts_preprocessed/font_img/{DATASET}'
    SAVE_DIR_NPZ = f'MyFonts_preprocessed/font_npz/{DATASET}'
    PREPROCESSED_FONT_FLAGS_PATH = f'MyFonts_preprocessed/font_flags_preprocessed/{DATASET}.csv'

    # create save folder
    os.makedirs(SAVE_DIR_IMG, exist_ok=True)
    os.makedirs(SAVE_DIR_NPZ, exist_ok=True)

    # read font flags
    font_flags_list = np.genfromtxt(PREPROCESSED_FONT_FLAGS_PATH, delimiter=',', dtype=str, skip_header=0)
    font_names = font_flags_list[:,0]
    font_flags = font_flags_list[:,1]
    font_num = len(font_names)

    # save preprocessed font as png and npz format
    p = pool.Pool(30)
    values = [[idx, font_names[idx], font_flags[idx]] for idx in range(font_num)]
    p.map(save_fonts, values)