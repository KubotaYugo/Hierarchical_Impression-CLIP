'''
font_flags_handcraft_{dataset}.csvをもとに, 実験に使用するフォント/使用しないフォントのフラグを作成
(文字は読めるが過度な装飾があるものでない and 文字でないものでない) or 大文字だけは使えそうなものを使用する

font_flags_handcraft_{dataset}.csv: 目視で以下のflagを立てたもの
    0列: フォント名
    1列: 文字は読めるが過度な装飾があるもの
    2列: 文字でないもの，抜けている文字があるもの
    3列: small-caps
        (完全に同じものもあれば，大きさが違うもの，一部だけのものや少しスタイルが違うものも)
    4列: 大文字だけは使えそうなもの
    5列: 小文字だけは使えそうなもの
    6列: 判断が微妙なもの(解決済み，このプログラムでは考慮しない)
'''


import numpy as np
import os
import utils_data_preprocessing as utils


# define parameters
args = utils.get_args()
DATASET = args.dataset
FONT_FLAGS_PATH = f'MyFonts_preprocessed/font_flags/{DATASET}.csv'
SAVE_PATH = f'MyFonts_preprocessed/font_flags_preprocessed/{DATASET}.csv'

# make directory to save font_flags_preprocessed
save_dir = os.path.dirname(SAVE_PATH)
os.makedirs(save_dir, exist_ok=True)

# load font flags
font_flags = np.genfromtxt(FONT_FLAGS_PATH, delimiter=',', dtype=str, skip_header=0)

# Preparing the data for saving
fontnames = font_flags[:,0]
preprocessed_font_flags = ((font_flags[:,1]=='0') * (font_flags[:,2]=='0')) + (font_flags[:,4]=='1')
preprocessed_font_flags = preprocessed_font_flags.astype(np.int8).astype('<U30')
save_data = list(zip(fontnames, preprocessed_font_flags))
np.savetxt(SAVE_PATH, save_data, delimiter=',', fmt='%s')