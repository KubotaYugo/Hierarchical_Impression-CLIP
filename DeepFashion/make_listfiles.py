"""
git(https://github.com/salesforce/hierarchicalContrastiveLearning?tab=readme-ov-file)で指定された方法で
パスとカテゴリをまとめる
 {
   "images": [
     "/deep_fashion_in_store/img/WOMEN/Dresses/id_00000002/02_1_front.jpg",
     "/deep_fashion_in_store/img/WOMEN/Dresses/id_00000002/02_2_side.jpg",
     "/deep_fashion_in_store/img/WOMEN/Dresses/id_00000002/02_4_full.jpg",
     "/deep_fashion_in_store/img/WOMEN/Dresses/id_00000002/02_7_additional.jpg",
     "/deep_fashion_in_store/img/WOMEN/Blouses_Shirts/id_00000004/03_1_front.jpg"
   ],
   "categories": [
     "Dresses",
     "Dresses",
     "Dresses",
     "Dresses",
     "Blouses_Shirts"
   ]
 }
"""
# "/myfiles/DeepFashion/In-shop Clothes Retrieval Benchmark/Img" 

import os
import json
import random
from collections import defaultdict


def is_image_file(filename):
    # 画像ファイルの拡張子を判定
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    return os.path.splitext(filename)[1].lower() in image_extensions

def get_image_paths_and_categories(directory):
    data_by_id = defaultdict(lambda: {"images": [], "categories": []})
    for root, _, files in os.walk(directory):
        for file in files:
            if is_image_file(file):
                file_path = os.path.join(root, file)
                id_ = file_path.split(os.sep)[-2]
                category = file_path.split(os.sep)[-3]
                data_by_id[id_]["images"].append(file_path)
                data_by_id[id_]["categories"].append(category)
    return data_by_id

def split_data(data_by_id, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    ids = list(data_by_id.keys())
    random.shuffle(ids)
    num_ids = len(ids)
    train_end = int(num_ids * train_ratio)
    val_end = train_end + int(num_ids * val_ratio)
    
    train_ids = ids[:train_end]
    val_ids = ids[train_end:val_end]
    test_ids = ids[val_end:]
    
    return (
        {id_: data_by_id[id_] for id_ in train_ids},
        {id_: data_by_id[id_] for id_ in val_ids},
        {id_: data_by_id[id_] for id_ in test_ids}
    )

def format_data(data):
    formatted_data = {"images": [], "categories": []}
    for id_data in data.values():
        formatted_data["images"].extend(id_data["images"])
        formatted_data["categories"].extend(id_data["categories"])
    return formatted_data

def save_to_json(data, json_file):
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main(directory, output_dir):
    random.seed(7)
    data_by_id = get_image_paths_and_categories(directory)
    train_data, val_data, test_data = split_data(data_by_id)
    
    os.makedirs(output_dir, exist_ok=True)
    
    save_to_json(format_data(train_data), os.path.join(output_dir, 'train.json'))
    save_to_json(format_data(val_data), os.path.join(output_dir, 'val.json'))
    save_to_json(format_data(test_data), os.path.join(output_dir, 'test.json'))
    
    print(f"Data has been split and saved to {output_dir}")


if __name__ == "__main__":
    target_directory = "dataset/img"   # 対象ディレクトリのパス
    output_json_file = 'dataset/listfile'  # jsonファイルを保存するディレクトリ
    main(target_directory, output_json_file)