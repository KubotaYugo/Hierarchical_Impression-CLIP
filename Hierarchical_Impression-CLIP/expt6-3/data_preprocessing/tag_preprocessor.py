'''
(1)文字化けしたタグとword2vecでencodeできないタグを除いて, オリジナルの頻度を取得
(2)各フォントが持つタグをtrain内の頻度順にソートし, top10に絞る
(3)タグを頻度top10に絞ったあとの頻度で並び替え
(4)top10に絞ったあとでtrain内頻度50未満のタグを除去
'''


import os
import numpy as np
from collections import Counter
import utils_data_preprocessing as utils

# 乱数の指定？

# define parameters
args = utils.get_args()
DATASET = args.dataset
INVALID_TAGS = utils.INVALID_TAGS
WORD2VEC_FLAG_PATH = 'MyFonts_preprocessed/word2vec_flag.csv'
TAG_FREQ_ORG_PATH = 'MyFonts_preprocessed/tag_freq_org.csv'
TAG_FREQ_TOP10_PATH = 'MyFonts_preprocessed/tag_freq_top10.csv'
PREPROCESSED_TAG_DIR = 'MyFonts_preprocessed/tag_txt'
FONTNAMES_USED_DIR = 'MyFonts_preprocessed/fontnames_used'


# (1) calculate the frequency of original tags (excluding corrupted tags and those not encodable with word2vec)
# load the flag indicating if word2vec encoding is possible
wrod2vec_flag = np.genfromtxt(WORD2VEC_FLAG_PATH, delimiter=',', dtype=str, skip_header=0)
word2vec_flag_dict = {row[0]: row[1] for row in wrod2vec_flag}

# calculate the frequency of original tags for each dataset
tag_freq_org = {}
for dataset in ['train', 'val', 'test']:
    fontnames = utils.get_font_name(dataset)
    tag_freq_org[dataset] = dict(
        Counter(
            tag
            for fontname in fontnames
            for tag in utils.get_original_tags(fontname)
            if (tag not in INVALID_TAGS) and (word2vec_flag_dict[tag]) != 0
        )
    )

# add tags included in other datasets to the dictionary with a frequency of 0
all_tag_list = set(tag_freq_org['train'].keys()) | set(tag_freq_org['val'].keys()) | set(tag_freq_org['test'].keys())
all_tag_list = sorted(all_tag_list)
for dataset in ['train', 'val', 'test']:
    for tag in all_tag_list:
        tag_freq_org[dataset].setdefault(tag, 0)

# save the frequency of original tags (order by frequency in the train set)
save_data = [['tag', 'total', 'train', 'val', 'test']]
all_tag_list_sorted = sorted(all_tag_list, key=lambda tag: tag_freq_org['train'][tag], reverse=True)
for tag in all_tag_list_sorted:
    total = tag_freq_org['train'][tag] + tag_freq_org['val'][tag] + tag_freq_org['test'][tag]
    row = [tag, total, tag_freq_org['train'][tag], tag_freq_org['val'][tag], tag_freq_org['test'][tag]]
    save_data.append(row)
np.savetxt(TAG_FREQ_ORG_PATH, save_data, delimiter=',', fmt='%s')



# (2) sort the tags of each font by frequency within the train set and keep the top 10
tags_dict_top10 = {'train':{}, 'val':{}, 'test':{}}
for dataset in ['train', 'val', 'test']:
    fontnames = utils.get_font_name(dataset)
    for fontname in fontnames:
        tags = utils.get_original_tags(fontname)
        tags = [tag for tag in tags if tag in all_tag_list]
        if tags:
            tags_dict_top10[dataset][fontname] = sorted(tags, key=all_tag_list_sorted.index)[:10]

# calculate the frequency of tags after removing tags beyond the top 10
tag_freq_top10 = {'train':{}, 'val':{}, 'test':{}}
for dataset in {'train', 'val', 'test'}:
    for tag in all_tag_list_sorted:
        tag_freq_top10[dataset][tag] = 0
    for tags in tags_dict_top10[dataset].values():
        for tag in tags:
            tag_freq_top10[dataset][tag]+=1



# (3) sort tags by frequency after filtering to the top 10
tag_list_top10_sorted = sorted(all_tag_list_sorted, key=lambda tag: tag_freq_top10['train'][tag], reverse=True)
tags_dict_top10_sorted = {'train':{}, 'val':{}, 'test':{}}
for dataset in ['train', 'val', 'test']:
    for fontname, tags in zip(tags_dict_top10[dataset].keys(), tags_dict_top10[dataset].values()):
        tags_dict_top10_sorted[dataset][fontname] = sorted(tags, key=tag_list_top10_sorted.index)

# save the tag frequency after filtering to the top 10
save_data = [['tag', 'total', 'train', 'val', 'test']]
for tag in tag_list_top10_sorted:
    total = tag_freq_top10['train'][tag]+tag_freq_top10['val'][tag]+tag_freq_top10['test'][tag]
    row = [tag, total, tag_freq_top10['train'][tag], tag_freq_top10['val'][tag], tag_freq_top10['test'][tag]]
    save_data.append(row)
np.savetxt(TAG_FREQ_TOP10_PATH, save_data, delimiter=',', fmt='%s')



# (4) save tags that have a frequency higher than 50 and saved font names
os.makedirs(FONTNAMES_USED_DIR, exist_ok=True)
used_font_names = {'train':[], 'val':[], 'test':[]} 
for dataset in ['train', 'val', 'test']:
    # make directory
    os.makedirs(f'{PREPROCESSED_TAG_DIR}/{dataset}', exist_ok=True)

    # save tags
    for fontname, tags in zip(list(tags_dict_top10_sorted[dataset].keys()), list(tags_dict_top10_sorted[dataset].values())):
        tags = [tag for tag in tags if tag_freq_top10['train'][tag] >= 50]
        if tags:
            np.savetxt(f'{PREPROCESSED_TAG_DIR}/{dataset}/{fontname}.csv', tags, delimiter=',', fmt='%s')
            used_font_names[dataset].append(fontname)
    
    # save font names
    np.savetxt(f'{FONTNAMES_USED_DIR}/{dataset}.csv', used_font_names[dataset], delimiter=',', fmt='%s')