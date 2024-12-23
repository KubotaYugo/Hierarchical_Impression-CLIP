from dotmap import DotMap
import argparse
import numpy as np


INVALID_TAGS = ['%d0%b3%d1%80%d0%be%d1%82%d0%b5%d1%81%d0%ba',
                '%d0%ba%d0%b8%d1%80%d0%b8%d0%bb%d0%bb%d0%b8%d1%86%d0%b0', 
                '%d9%86%d8%b3%d8%ae', 
                '']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    args = vars(args)
    args = DotMap(args)
    return args


def get_font_name(dataset):
    '''
    フラグが1のフォントのリストを返す
    '''
    PREPROCESSED_FONT_FLAGS_PATH = f'MyFonts_preprocessed/font_flags_preprocessed/{dataset}.csv'
    font_flags_list = np.genfromtxt(PREPROCESSED_FONT_FLAGS_PATH, delimiter=',', dtype=str, skip_header=0)
    font_names = font_flags_list[:,0]
    font_flags = font_flags_list[:,1]
    return font_names[font_flags=='1']


def get_original_tags(font_name):
    tag_path = f'MyFonts/taglabel/{font_name}'
    tags = np.genfromtxt(tag_path, delimiter=' ', dtype=str, skip_header=0)
    if tags.shape==():
        return [tags.item()]
    return tags.tolist()