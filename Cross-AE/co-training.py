import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import os
import matplotlib.pyplot as plt
import random
import numpy as np
import copy
import csv
from dataloader import DataLoader
import models
import tools
import FontAutoencoder


def PlotLoss(epoch, train_loss_list, valid_loss_list, SAVE_PATH, name):
    plt.figure(figsize=(10,10))
    plt.plot(range(1, epoch+1), train_loss_list)
    plt.plot(range(1, epoch+1), valid_loss_list)
    plt.xlim(1, epoch)
    plt.ylim(0, 0.01)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'validation'])
    plt.savefig(f"{SAVE_PATH}/{name}", dpi=500)
    plt.close()
    
def SaveLossCSV(epoch, train_loss_list, valid_loss_list, SAVE_PATH, name):
    f = open(f"{SAVE_PATH}/{name}.csv", 'w')
    writer = csv.writer(f)
    for e in range(1, epoch+1):
        writer.writerow([e, train_loss_list[e-1], valid_loss_list[e-1]])
    f.close()
    
    
    
    
    

# ハイパラ
LEARNING_RATE = 0.001
BATCH_SIZE = 128
patience = 100
EPOCH = 5000

# 保存用ディレクトリの作成
SAVE_PATH = "Cross-AE/co-training"
os.makedirs(f"{SAVE_PATH}", exist_ok=True)
os.makedirs(f"{SAVE_PATH}/imp-AE", exist_ok=True)
os.makedirs(f"{SAVE_PATH}/img-AE", exist_ok=True)

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
dataset_train = tools.FontDataloader2(img_dir1, tag_dir1, label_dir)
dataset_valid = tools.FontDataloader2(img_dir2, tag_dir2, label_dir)
NUM_WORKERS = 0     #ここを0にしないと動かないかも
loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
loader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# モデル, オプティマイザ, ロスの準備
device = 'cuda'
# img-AE
model_img = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device) 
FONT_AUTOENCODER_STATE_PATH = f"FontAutoencoder/model/best.pt" 
model_img.load_state_dict(torch.load(FONT_AUTOENCODER_STATE_PATH))
# imp-AE
model_imp = models.IntegratedModel(models.DeepSets(), models.imp_Encoder(), models.imp_Decoder()).to(device)
model_path = 'Cross-AE/imp-AE/model/best(epoch=122).pt'
state_dict = torch.load(model_path, map_location='cuda:0')
model_imp.load_state_dict(state_dict)
# optimizer, loss
parameters = list(model_img.parameters())+list(model_imp.parameters())
optimizer = optim.Adam(parameters, lr=LEARNING_RATE)
criterion = nn.MSELoss()



# train&validation
train_loss_list = []
train_loss1_list = []
train_loss2_list = []
train_loss3_list = []
train_loss4_list = []
valid_loss_list = []
valid_loss1_list = []
valid_loss2_list = []
valid_loss3_list = []
valid_loss4_list = []
bad = 0
val_loss_best = np.Inf
for epoch in tqdm(range(1, EPOCH+1)):
    print('epoch', epoch)
    train_loss = 0
    train_loss1 = 0
    train_loss2 = 0
    train_loss3 = 0
    model_imp.train()
    model_img.train()
    for data in loader_train:
        image, w2v, w2v_mean = data["image"], data["w2v"], data["w2v_mean"]
        # 印象側
        w2v_mean = w2v_mean.to(device)
        _,imp_feature,output2 = model_imp(w2v)
        # 画像側
        image = image.to(device)
        # grd_output, dec_output = model_img(image)
        grd_output = model_img.encoder(image)
        dec_output = model_img.decoder(grd_output)
        # ロスの計算
        loss1 = criterion(grd_output, imp_feature)
        loss2 = criterion(image, dec_output)
        loss3 = criterion(output2, w2v_mean)
        loss = loss1 + loss2 + loss3
        train_loss += loss.item()
        train_loss1 += loss1.item()
        train_loss2 += loss2.item()
        train_loss3 += loss3.item()
        # パラメータ更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_train_loss = train_loss / len(loader_train)
    avg_train_loss1 = train_loss1 / len(loader_train)
    avg_train_loss2 = train_loss2 / len(loader_train)
    avg_train_loss3 = train_loss3 / len(loader_train)
    train_loss_list.append(avg_train_loss)
    train_loss1_list.append(avg_train_loss1)
    train_loss2_list.append(avg_train_loss2)
    train_loss3_list.append(avg_train_loss3)
    
    
    
    # validation
    valid_loss = 0
    valid_loss1 = 0
    valid_loss2 = 0
    valid_loss3 = 0
    model_imp.eval()
    model_img.eval()
    for data in loader_valid:
        with torch.no_grad():
            image, w2v, w2v_mean = data["image"], data["w2v"], data["w2v_mean"]
            # 印象側
            w2v_mean = w2v_mean.to(device)
            _,imp_feature,output2 = model_imp(w2v)
            # 画像側
            image = image.to(device)
            grd_output = model_img.encoder(image)
            dec_output = model_img.decoder(grd_output)
            # ロスの計算
            loss1 = criterion(grd_output, imp_feature)
            loss2 = criterion(image, dec_output)
            loss3 = criterion(output2, w2v_mean)
            loss = loss1 + loss2 + loss3
            valid_loss += loss.item()
            valid_loss1 += loss1.item()
            valid_loss2 += loss2.item()
            valid_loss3 += loss3.item()
    avg_valid_loss = valid_loss / len(loader_valid)
    avg_valid_loss1 = valid_loss1 / len(loader_valid)
    avg_valid_loss2 = valid_loss2 / len(loader_valid)
    avg_valid_loss3 = valid_loss3 / len(loader_valid)
    valid_loss_list.append(avg_valid_loss)
    valid_loss1_list.append(avg_valid_loss1)
    valid_loss2_list.append(avg_valid_loss2)
    valid_loss3_list.append(avg_valid_loss3)
    
    # ロスの表示
    print('epoch: {}, train_loss: {:.7f}, val_loss: {:.7f}'.format(epoch, avg_train_loss, avg_valid_loss))
    print('train_pair_loss: {:.7f}, train_img_mse_loss: {:.7f}, train_imp_mse_loss: {:.7f},'
          'valid_pair_loss: {:.7f}, valid_img_mse_loss: {:.7f},  valid_imp_mse_loss: {:.7f}'.
          format(avg_train_loss1, avg_train_loss2, avg_train_loss3, avg_valid_loss1, avg_valid_loss2, avg_valid_loss3))
    
    # Early Stoping
    if avg_valid_loss < val_loss_best:
        print('Validation loss decreased ({:.7f} --> {:.7f})'.format(val_loss_best, avg_valid_loss))
        # lossを更新したときのモデルをコピー
        model_imp_best = copy.deepcopy(model_imp.state_dict())
        model_img_best = copy.deepcopy(model_img.state_dict())
        bad = 0
        val_loss_best = avg_valid_loss
        epoch_best = epoch
    else:
        bad += 1
        print(f'EarlyStopping counter: {bad} out of {patience}')
    if bad == patience:
        break

# bestなモデルの保存
torch.save(model_imp_best, f"{SAVE_PATH}/imp-AE/best(epoch={epoch_best}).pt")
torch.save(model_img_best, f"{SAVE_PATH}/img-AE/best(epoch={epoch_best}).pt")



# 結果の出力
# lossのグラフを保存
PlotLoss(epoch, train_loss_list, valid_loss_list, SAVE_PATH, "loss")
PlotLoss(epoch, train_loss1_list, valid_loss1_list, SAVE_PATH, "loss1")
PlotLoss(epoch, train_loss2_list, valid_loss2_list, SAVE_PATH, "loss2")
PlotLoss(epoch, train_loss3_list, valid_loss3_list, SAVE_PATH, "loss3")
# csvに保存
SaveLossCSV(epoch, train_loss_list, valid_loss_list, SAVE_PATH, "loss")
SaveLossCSV(epoch, train_loss1_list, valid_loss1_list, SAVE_PATH, "loss1")
SaveLossCSV(epoch, train_loss2_list, valid_loss2_list, SAVE_PATH, "loss2")
SaveLossCSV(epoch, train_loss3_list, valid_loss3_list, SAVE_PATH, "loss3")