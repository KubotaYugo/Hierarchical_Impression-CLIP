import torch
import glob
import cv2
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset
from pathlib import Path
import random

random.seed(1)


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


class FontDataloader(Dataset):
    def __init__(self, img_dir, tag_dir, label_dir):
        self.imgs_paths = Path(img_dir)
        self.font_list = sorted(glob.glob(tag_dir + "/*"))
        self.label_dir = Path(label_dir)
        self.label_list = []
        self.label2_list = []
        self.make_label()

    def make_label(self):
        for i in range(len(self.font_list)):
            img_name = self.imgs_paths.joinpath(self.font_list[i])
            label_path = self.label_dir.joinpath(img_name.stem)  # fontの名前ぶんとばす
            w2v = torch.from_numpy(np.load(f"{label_path}.npz")["arr_0"])
            self.label_list.append(w2v)
            label = w2v.mean(0)
            self.label2_list.append(label)

    def __len__(self):
        return len(self.font_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'w2v': self.label_list[idx], 'w2v_mean': self.label2_list[idx]}

        return sample


class FontDataloader2(Dataset):
    def __init__(self, img_dir, tag_dir, label_dir):

        self.imgs_paths = Path(img_dir)
        self.font_list = sorted(glob.glob(tag_dir + "/*"))
        self.label_dir = Path(label_dir)
        self.label_list = []
        self.label2_list = []
        self.make_label()

    def make_label(self):
        for i in range(len(self.font_list)):
            img_name = self.imgs_paths.joinpath(self.font_list[i])
            label_path = self.label_dir.joinpath(img_name.stem)
            w2v = torch.from_numpy(np.load(f"{label_path}.npz")["arr_0"])
            self.label_list.append(w2v)
            label = w2v.mean(0)
            self.label2_list.append(label)

    def __len__(self):
        return len(self.font_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = str(self.imgs_paths)+"/"+str(self.imgs_paths.joinpath(self.font_list[idx]).stem)+".npz"
        images = np.load(img_path)["arr_0"].astype(np.float32)
        images = torch.from_numpy(images/255)

        sample = {'image': images, 'w2v': self.label_list[idx], 'w2v_mean': self.label2_list[idx]}

        return sample


class FontDataloader3(Dataset):
    def __init__(self, img_dir, label_dir, impression, target_dir):

        self.imgs_paths = Path(img_dir)
        self.font_list = sorted(glob.glob(img_dir + "/*"))
        self.label_dir = Path(label_dir)
        self.target_dir = Path(target_dir)
        self.label_list = []
        self.label2_list = []
        self.mark_list = []
        self.imp = impression
        self.make_label()

    def make_label(self):
        for i in range(len(self.font_list)):
            img_name = self.imgs_paths.joinpath(self.font_list[i])
            w2v = str(img_name.stem)
            self.label2_list.append(w2v)
            self.label_list.append(w2v)
            target_path = self.target_dir.joinpath(img_name.stem)
            with open(target_path, 'r') as f:
                labels = f.read()
                labels = labels.split('\n')[:-1]
            cnt = 0
            for word in labels:
                if word == self.imp:
                    cnt = 1
                    break
            if cnt == 0:
                self.mark_list.append('x')
            else:
                self.mark_list.append('O')

    def __len__(self):
        return len(self.font_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        images = torch.load(self.font_list[idx])

        sample = {'image': images, 'w2v': self.label_list[idx], 'w2v_mean': self.label2_list[idx], 'mark': self.mark_list[idx]}

        return sample


# png to numpy (resize 224 x 224)
def word2char(img_path, img_size):
    # 画像の読み込み
    img_gray = cv2.imread(img_path, 0)

    # 大津の２値化
    _thre, img_th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # オブジェクト検出する際に白い箇所を検出するので，白黒反転の処理
    img_chara = np.where(img_th != 0, 0, 255)
    img_chara = img_chara.astype(np.uint8)

    # オブジェクト検出（文字領域検出）
    contours, hierarchy = cv2.findContours(img_chara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = np.shape(img_chara)

    # contoursの中に入っている座標(オブジェクトの座標)をすべてチェックして，最小の座標値を取得
    # ここ要注意！！ここで得られる(x,y)は座標，画像リサイズのときは行列のインデックス
    area = []
    min_x = []
    min_y = []
    max_x = []
    max_y = []
    for i in range(0, len(contours)):
        p = (contours[i]).reshape((-1, 2))
        min_x.append(np.min(p[:, 1]))
        min_y.append(np.min(p[:, 0]))
        max_x.append(np.max(p[:, 1]))
        max_y.append(np.max(p[:, 0]))

    x = np.min(min_x)
    y = np.min(min_y)
    h = np.max(max_x)
    w = np.max(max_y)

    size = max(h, w)
    ratio = img_size / size

    # ここのx,yは行列インデックス
    # img_resize = cv2.resize(img_chara[x:h,y:w],(img_size,img_size))
    img_resize = cv2.resize(img_chara[x:h, y:w], (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_CUBIC)

    # 0埋めの幅を決める
    if w > h:
        pad = int((img_size - h * ratio) / 2)
        # np.pad()の第二引数[(上，下),(左，右)]にpaddingする行・列数
        img_resize = np.pad(img_resize, [(pad, pad), (0, 0)], 'constant')
    elif h > w:
        pad = int((img_size - w * ratio) / 2)
        img_resize = np.pad(img_resize, [(0, 0), (pad, pad)], 'constant')

    # 最終的にきれいに100x100にresize
    img_resize = cv2.resize(img_resize, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    img_resize = np.where(img_resize != 0, 0, 255)
    img_resize = img_resize / 255

    return img_resize
