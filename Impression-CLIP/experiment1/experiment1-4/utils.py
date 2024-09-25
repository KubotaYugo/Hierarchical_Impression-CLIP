import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
from pprint import pprint
import numpy as np
import csv
import random
import math

EXP = "Impression-CLIP/experiment1/experiment1-4"
BATCH_SIZE = 1024
    
class ExpMultiplier(nn.Module):
    def __init__(self, initial_value=0.0):
        super(ExpMultiplier, self).__init__()
        self.t = nn.Parameter(torch.tensor(initial_value, requires_grad=True))
    def forward(self, x):
        return x * torch.exp(self.t)
    
class EarlyStopping:
    def __init__(self, patience, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.early_stop = False
        self.min_val_loss = np.Inf
    def __call__(self, val_loss):
        if self.min_val_loss+self.delta <= val_loss:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            if  val_loss < self.min_val_loss:
                print(f'Validation loss decreased ({self.min_val_loss} --> {val_loss})')
                self.min_val_loss = val_loss



def fix_seed(seed):
    """
    乱数の固定
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm



def LoadDatasetPaths(dataset):
    """
    datasetのフォントとタグのパスのリストを返す
    """
    with open(f"dataset/MyFonts_preprocessed/tag_txt/fontname/{dataset}.csv") as f:
        reader = csv.reader(f)
        font_names = np.asarray([row for row in reader])
    image_paths = [f"dataset/MyFonts_preprocessed/font_numpy_Impression-CLIP/{dataset}/{font_name[0]}.npz" for font_name in font_names]
    tag_paths = [f"dataset/MyFonts_preprocessed/tag_txt/{dataset}/{font_name[0]}.csv" for font_name in font_names]
    return image_paths, tag_paths




class CustomDataset(Dataset):
    def __init__(self, font_paths, tag_paths, tokenizer):
        self.font_paths = font_paths
        self.tag_paths = tag_paths
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.font_paths)
    def __getitem__(self, idx):
        #----------フォント----------
        font = np.load(self.font_paths[idx])["arr_0"].astype(np.float32)
        font = torch.from_numpy(font/255)
        #----------タグ----------
        with open(self.tag_paths[idx], encoding='utf8') as f:
            csvreader = csv.reader(f)
            tags = [row for row in csvreader][0]
        if len(tags) == 1:
            prompt = f"The impression is {tags[0]}."
        elif len(tags) == 2:
            prompt = f"First and second impressions are {tags[0]} and {tags[1]}, respectively."
        elif len(tags) >= 3:
            ordinal = ["First", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
            prompt1 = ordinal[0]
            prompt2 = tags[0]
            i = 0
            for i in range(1, min(len(tags)-1, 10-1)):
                prompt1 = prompt1 + ", " + ordinal[i]
                prompt2 = prompt2 + ", " + tags[i]
            prompt1 = prompt1 + ", and " + ordinal[i+1] + " impressions are "
            prompt2 = prompt2 + ", and " + tags[i+1] + ", respectively."                
            prompt = prompt1 + prompt2
        tokenized_text = self.tokenizer(prompt, return_tensors="pt", max_length=self.tokenizer.max_model_input_sizes['openai/clip-vit-base-patch32'], padding="max_length", truncation=True)
        return font, tokenized_text['input_ids'][0]

    

def train(epoch, models, optimizer, lr_scheduler, criterion, dataloader, logger, device):
    # モデルをほぐす
    font_encoder = models[0] 
    clip_model = models[1]
    temp = models[2]
    
    # trainに
    font_encoder.train()
    clip_model.train()
    temp.train()
    
    running_loss = []
    for i, data in enumerate(dataloader):
        # 勾配の初期化
        optimizer.zero_grad()

        # fontとtagをモデルに通す
        font = Variable(data[0]).to(device)
        tokenized_text = Variable(data[1]).to(device)
        font_feature = font_encoder(font)
        text_feature = clip_model.get_text_features(tokenized_text)
        font_feature = F.normalize(font_feature, dim=1)
        text_feature = F.normalize(text_feature, dim=1)

        # 画像特徴・印象特徴の類似度の行列を計算
        matrix = torch.matmul(font_feature, text_feature.T)
        logits = temp(matrix)   #温度パラメータの適用

        # ロスの計算
        labels = torch.eye(logits.shape[0]).to(device)
        loss_f = criterion(logits.T, labels)
        loss_t = criterion(logits, labels)
        loss = (loss_f+loss_t)/2
        running_loss.append(loss.item())

        # 学習率の保存
        logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # backward
        loss.backward()
        optimizer.step()   
    lr_scheduler.step()
            
    total_batch_loss = np.asarray(running_loss).mean()
    pprint(f"[{epoch}] Epoch: {epoch}, loss: {total_batch_loss}.")
    
    return total_batch_loss



def val(epoch, models, criterion, dataloader, device):
    # モデルをほぐす
    font_encoder = models[0] 
    clip_model = models[1]
    temp = models[2]
    
    # evalに
    font_encoder.eval()
    clip_model.eval()
    temp.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            #----------fontとtagをモデルに通す----------
            font = Variable(data[0]).to(device)
            tokenized_text = Variable(data[1]).to(device)
            font_feature = font_encoder(font)
            text_feature = clip_model.get_text_features(tokenized_text)
            font_feature = F.normalize(font_feature, dim=1)
            text_feature = F.normalize(text_feature, dim=1)
            if i==0:
                font_features = font_feature
                text_features = text_feature
            else:
                font_features = torch.cat([font_features, font_feature])
                text_features = torch.cat([text_features, text_feature])  

    #----------画像特徴・印象特徴の類似度の行列を計算----------
    matrix = torch.matmul(font_features, text_features.T)
    logits = temp(matrix)

    #----------ロスの計算----------
    labels = torch.eye(logits.shape[0]).to(device)
    loss_i = criterion(logits.T, labels)
    loss_t = criterion(logits, labels)
    loss = (loss_i+loss_t)/2
    
    #----------温度なしのロスを計算----------
    with torch.no_grad():
        labels = torch.eye(logits.shape[0]).to(device)
        loss_without_temp_i = criterion(matrix.T, labels)
        loss_without_temp_t = criterion(matrix, labels)
        loss_without_temp = (loss_without_temp_i+loss_without_temp_t)/2
    #----------ロスの表示----------
    pprint(f"[{epoch}] Epoch: {epoch}, loss: {loss.item()}.")
    pprint(f"[{epoch}] Epoch: {epoch}, loss without temp: {loss_without_temp.item()}.")
        
    return loss.item(), loss_without_temp.item(), temp.t.item()


# 使ってない
# def adjust_learning_rate(args, optimizer, epoch):
#     lr = args.learning_rate
#     if args.cosine:
#         eta_min = lr * (args.lr_decay_rate ** 3)
#         lr = eta_min + (lr-eta_min)*(1+math.cos(math.pi*epoch/args.epochs))/2
#     else:
#         steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
#         if steps > 0:
#             lr = lr * (args.lr_decay_rate ** steps)

#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

# 使ってない
# def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
#     if args.warm and epoch <= args.warm_epochs:
#         p = (batch_id+(epoch-1)*total_batches) / (args.warm_epochs*total_batches)
#         lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr