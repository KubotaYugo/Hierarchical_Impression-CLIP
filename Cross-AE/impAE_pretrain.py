import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
import random
import numpy as np
import csv
import copy
from dataloader import DataLoader
import models
import tools
import os





#-----------ハイパラ-----------
LEARNING_RATE = 0.001
BATCH_SIZE = 128
patience = 100
EPOCH = 10000

# 保存用ディレクトリの作成
SAVE_PATH = f"Cross-AE/imp-AE"
os.makedirs(f"{SAVE_PATH}/model", exist_ok=True)

# 乱数の固定
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)  # fix the initial value of the network weight
torch.cuda.manual_seed(seed)  # for cuda
torch.cuda.manual_seed_all(seed)  # for multi-GPU
torch.backends.cudnn.deterministic = True  # choose the determintic algorithm



# dataloder
img_dir1 = "dataset/MyFonts_CompWithCrossAE/font_numpy/train"
img_dir2 = "dataset/MyFonts_CompWithCrossAE/font_numpy/val"
tag_dir1 = "dataset/MyFonts_CompWithCrossAE/tag_txt/train"
tag_dir2 = "dataset/MyFonts_CompWithCrossAE/tag_txt/val"
label_dir = "Cross-AE/impression-vector"
dataset_train = tools.FontDataloader(img_dir1, tag_dir1, label_dir)
dataset_valid = tools.FontDataloader(img_dir2, tag_dir2, label_dir)
NUM_WORKERS = 0     # 0じゃないと動かない
loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
loader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# モデル, オプティマイザ, ロスの準備
device = 'cuda'
model_DS = models.IntegratedModel(models.DeepSets(), models.imp_Encoder(), models.imp_Decoder()).to(device)
optimizer2 = optim.Adam(model_DS.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()



# train&validation
train_loss_list = []
valid_loss_list = []
bad = 0
val_loss_best = np.Inf
for epoch in tqdm(range(1, EPOCH+1)):
    print('epoch', epoch)

    # train
    train_loss = 0
    model_DS.train()
    for data in loader_train:
        w2v, w2v_label = data["w2v"], data["w2v_mean"]
        w2v_label = w2v_label.to(device)
        _,_,grd_label = model_DS(w2v)
        loss = criterion(grd_label, w2v_label)
        train_loss += loss.item()
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
    avg_train_loss = train_loss / len(loader_train)
    train_loss_list.append(avg_train_loss)

    # validation
    valid_loss = 0
    model_DS.eval()
    for data in loader_valid:
        with torch.no_grad():
            w2v, w2v_label = data["w2v"], data["w2v_mean"]
            w2v_label = w2v_label.to(device)
            _,_,grd_label = model_DS(w2v)
            loss = criterion(grd_label, w2v_label)
            valid_loss += loss.item()
    avg_valid_loss = valid_loss / len(loader_valid)
    valid_loss_list.append(avg_valid_loss)
    
    print("epoch: {}, train_loss: {:.7f}, val_loss: {:.7f}".format(epoch, avg_train_loss, avg_valid_loss))
    
    # 100epochs毎にモデルを保存
    if epoch%100==0:
        torch.save(model_DS.state_dict(), SAVE_PATH+"/model/checkpoint_{:04d}.pt".format(epoch))
    
    # Early Stoping
    if avg_valid_loss < val_loss_best:
        print('Validation loss decreased ({:.7f} --> {:.7f})'.format(val_loss_best, avg_valid_loss))
        # lossを更新したときのモデルをコピ
        best_epoch = epoch
        model_DS_best = copy.deepcopy(model_DS.state_dict())
        bad = 0
        val_loss_best = avg_valid_loss
    else:
        bad += 1
        print(f'EarlyStopping counter: {bad} out of {patience}')
    if bad == patience:
        break
# bestなモデルの保存
torch.save(model_DS_best, f"{SAVE_PATH}/model/best(epoch={best_epoch}).pt")



# 結果の出力
# lossのグラフを保存
plt.figure(figsize=(10,10))
plt.plot(range(1, epoch+1), train_loss_list)
plt.plot(range(1, epoch+1), valid_loss_list, c='#00ff00')
plt.xlim(1, epoch)
plt.ylim(0, 0.005)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'validation'])
plt.savefig(f"{SAVE_PATH}/loss.png", dpi=500)
plt.close()
# lossをcsvに保存
f = open(f"{SAVE_PATH}/loss.csv", 'w')
writer = csv.writer(f)
for e in range(1, epoch+1):
    writer.writerow([e, train_loss_list[e-1], valid_loss_list[e-1]])
f.close()