"""
jihunさんの方法に合わせたタグの前処理
各フォントが持つ印象語をw2vでベクトルにして, 印象語数*300(ベクトルの次元)のnumpy配列で保存
datasetごとにディレクトリを分けずに, 全フォント同じディレクトリに保存
(1)フォントが持つタグの読み込み
(2)word2vecでencode
(3)重ねて(フォントが持つ印象語の数, 300)次元のnumpy配列で保存
"""
import csv
import numpy as np
import gensim
import sys
import os


os.makedirs('Cross-AE/impression-vector', exist_ok=True)

# word2vecの読み込み
word2vec = gensim.models.KeyedVectors.load_word2vec_format('Cross-AE/GoogleNews-vectors-negative300.bin', binary=True)


# encodeフラグの読み込み
with open(f"Cross-AE/word2vec_check.csv") as csv_f:
    reader = csv.reader(csv_f)
    rows = np.array([row for row in reader])
tag_list = rows[:,0]
encode_flag = np.array(rows[:,2], dtype=np.int32)

encode_flag_dict = {}
for i in range(len(tag_list)):
    encode_flag_dict[tag_list[i]] = encode_flag[i]



for dataset in ["train", "val", "test"]:
    #----------使用するフォント一覧の取得----------
    with open(f"dataset/MyFonts_CompWithCrossAE/tag_txt/fontname_{dataset}.csv") as csv_f:
        reader = csv.reader(csv_f)
        font_names = np.array([row[0] for row in reader])
    
    for font_name in font_names:
        print(f"{dataset}: {font_name}")
        #-----------タグの読み込み----------
        with open(f"dataset/MyFonts_CompWithCrossAE/tag_txt/{dataset}/{font_name}.csv") as csv_f:
            reader = csv.reader(csv_f)
            tags = np.array([row for row in reader])[0]
        
        #-----------タグのencode----------
        w2v_stack = []
        for tag in tags:
            if encode_flag_dict[tag]==1:
                w2v = word2vec[tag]
            elif encode_flag_dict[tag]==2:
                w2v = word2vec[tag.replace('-', '_')]
            elif encode_flag_dict[tag]==3:
                w2v = 0
                for i, split in enumerate(tag.split('-')):
                    w2v += word2vec[split]
                w2v = w2v/(i+1)
            else:
                print("error: encode_flag==0")
                sys.exit(1)
            w2v_stack.append(w2v)
        w2v_stack = np.asarray(w2v_stack)
        np.savez_compressed(f"Cross-AE/impression-vector/{font_name}.npz", w2v_stack)