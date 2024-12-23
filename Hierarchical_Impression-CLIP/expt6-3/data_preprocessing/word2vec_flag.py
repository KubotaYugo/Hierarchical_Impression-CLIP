'''
MyFontsデータセット内の印象語について, 以下をまとめる．
w2vでencodeできるかできないか
    0: できない
    1: できる(単語単体)
    2: できる(n-gramのハイフンをアンダーバーに変えればできる)
    3: できる(n-gramを分割して平均)
'''


import gensim
import numpy as np
import utils_data_preprocessing as utils
from natsort import natsorted


# define parameters
INVALID_TAGS = utils.INVALID_TAGS
WORD2VEC_PATH = f'GoogleNews-vectors-negative300.bin'
WORD2VEC_FLAG_PATH = 'MyFonts_preprocessed/word2vec_flag.csv'

# load word2vec feature
word2vec = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)
vocab = natsorted(word2vec.index_to_key)

# collect all impression words
tag_list = []
for dataset in ['train', 'val', 'test']:
    font_name_list = utils.get_font_name(dataset)
    for font_name in font_name_list:
        tags = utils.get_original_tags(font_name)
        tag_list = set(list(tag_list)+tags)
tag_list = sorted(list(tag_list))


# check_word2vec_support_for_impression_words
# 1: the tag can be encoded alone
# 2: encoding the n-gram is possible after replacing hyphens with underscores
# 3: the n-gram can be encoded after splitting
# 0: the tag can't be encode 
vocab_flag = {}
for tag in tag_list:
    if ('-' not in tag) and (tag in vocab):
        vocab_flag[tag] = 1
    elif tag.replace('-', '_') in vocab:
        vocab_flag[tag] = 2
    elif all(split in vocab for split in tag.split('-')):    
        vocab_flag[tag] = 3
    else:
        vocab_flag[tag] = 0

# save to csv
save_data = list(zip(vocab_flag.keys(), vocab_flag.values()))
np.savetxt(WORD2VEC_FLAG_PATH, save_data, delimiter=',', fmt='%s')