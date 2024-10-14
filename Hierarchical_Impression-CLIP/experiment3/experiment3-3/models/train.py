from transformers import CLIPTokenizer, CLIPModel
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import losses
import numpy as np
import csv
import wandb
import argparse
from DMH import DMH_D, DMH_BS
import FontAutoencoder
import MLP

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils

class EarlyStopping:
    def __init__(self, patience, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.early_stop = False
        self.min_value = np.Inf
    def __call__(self, value):
        if self.min_value+self.delta <= value:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            if  value < self.min_value:
                print(f'Validation meanARR decreased ({self.min_value} --> {value})')
                self.min_value = value

def train(dataloader, models, criterion, optimizer, total_itr, save_folder):
    # one epoch training
    font_autoencoder, clip_model, emb_i, emb_t = models
    font_autoencoder.eval()
    clip_model.eval()
    emb_i.train()
    emb_t.train()

    # Iterate over data
    # サンプリング方法の影響で, iteration数はepochごとに変わる(少しだけ)
    # また，1epochで同じデータが複数回でてくることもある
    loss_img_list = []
    loss_tag_list = []
    loss_list = []
    for idx, data in enumerate(dataloader):
        imgs, tokenized_tags, img_labels, tag_labels = data
        imgs, tokenized_tags, img_labels, tag_labels = imgs[0], tokenized_tags[0], img_labels[0], tag_labels[0]
        imgs = imgs.cuda(non_blocking=True)
        tokenized_tags = tokenized_tags.cuda(non_blocking=True)
        img_labels = img_labels.cuda(non_blocking=True)
        tag_labels = tag_labels.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            img_features = font_autoencoder.encoder(imgs)
            tag_features = clip_model.get_text_features(tokenized_tags)
        with torch.set_grad_enabled(True):
            # get model outputs and calculate loss       
            embedded_img_features = emb_i(img_features)
            embedded_tag_features = emb_t(tag_features)
            features = torch.cat([embedded_img_features.unsqueeze(1), embedded_tag_features.unsqueeze(1)], dim=1)
            # culuculate loss
            loss_img, layer_loss_list_img = criterion(features, img_labels)
            loss_tag, layer_loss_list_tag = criterion(features, tag_labels)
            loss = (loss_img+loss_tag)/2
            loss_img_list.append(loss_img.item())
            loss_tag_list.append(loss_tag.item())
            loss_list.append(loss.item())
            # backward + optimize only if in training phase
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        # save results to wandb
        save_dict = {'loss_img_itr': loss_img.item(),
                     'loss_tag_itr': loss_tag.item(),
                     'loss_itr': loss.item()
                    }
        for i in range(len(layer_loss_list_img)):
            save_dict[f'loss_img_itr_layer{len(layer_loss_list_img)-i}'] = layer_loss_list_img[i]
            save_dict[f'loss_tag_itr_layer{len(layer_loss_list_img)-i}'] = layer_loss_list_tag[i]
        wandb.log(save_dict, step=total_itr)
    
        # save results to csv file
        f = open(f"{save_folder}/result_itr.csv", 'a')
        csv_writer = csv.writer(f)
        csv_writer.writerow([total_itr, loss_img.item(), loss_tag.item(), loss.item()]+layer_loss_list_img+layer_loss_list_tag)
        f.close()

        total_itr += 1
    
    return np.mean(loss_img_list), np.mean(loss_tag_list), np.mean(loss_list), total_itr

def retrieval_rank(similarity_matrix, mode=None):
    if mode=="tag2img":
        similarity_matrix = similarity_matrix.T
    sorted_index = torch.argsort(similarity_matrix, dim=1, descending=True)
    rank = [torch.where(sorted_index[i]==i)[0].item()+1 for i in range(sorted_index.shape[0])]
    return rank

def val(dataloader, models):
    # one epoch validation
    font_autoencoder, clip_model, emb_i, emb_t = models
    font_autoencoder.eval()
    clip_model.eval()
    emb_i.eval()
    emb_t.eval()

    # extruct embedded features
    for idx, data in enumerate(dataloader):
        imgs, tokenized_tags = data
        imgs = imgs.cuda(non_blocking=True)
        tokenized_tags = tokenized_tags.cuda(non_blocking=True)
        with torch.no_grad():
            img_features = font_autoencoder.encoder(imgs)
            tag_features = clip_model.get_text_features(tokenized_tags) 
            embedded_img_features = emb_i(img_features)
            embedded_tag_features = emb_t(tag_features)
        if idx==0:
            embedded_img_features_stack = embedded_img_features
            embedded_tag_features_stack = embedded_tag_features
        else:
            embedded_img_features_stack = torch.concatenate((embedded_img_features_stack, embedded_img_features), dim=0)
            embedded_tag_features_stack = torch.concatenate((embedded_tag_features_stack, embedded_tag_features), dim=0)
    similarity_matrix = torch.matmul(embedded_img_features_stack, embedded_tag_features_stack.T)
    # culculate Average Precision
    ARR_tag2img = np.mean(retrieval_rank(similarity_matrix, "tag2img"))
    ARR_img2tag = np.mean(retrieval_rank(similarity_matrix, "img2tag"))
    meanARR = (ARR_img2tag+ARR_tag2img)/2
    
    return ARR_tag2img, ARR_img2tag, meanARR


if __name__ == '__main__':
    # define constant
    EXP = utils.EXP
    IMG_HIERARCHY_PATH = utils.IMG_HIERARCHY_PATH
    TAG_HIERARCHY_PATH = utils.TAG_HIERARCHY_PATH
    MODEL_PATH = utils.MODEL_PATH
    MAX_EPOCH = utils.MAX_EPOCH
    EARLY_STOPPING_PATIENCE = utils.EARLY_STOPPING_PATIENCE

    # parameter from args
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float)
    parser.add_argument('--bs', type=int)
    args = parser.parse_args()
    LR = args.lr
    BATCH_SIZE = args.bs

    # make save directory
    SAVE_FOLDER = f'{EXP}/LR={LR}, BS={BATCH_SIZE}/results'
    save_folder = SAVE_FOLDER + '/model'
    os.makedirs(save_folder, exist_ok=True)

    # set WandB
    wandb.init(
    project = "HIC3-3",
    name = f'LR={LR}, BS={BATCH_SIZE}',
    config = {
    "learning_rate": LR,
    "batch_size": BATCH_SIZE,
    "max_epoch": MAX_EPOCH,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE
    }
    )

    # get args, fix random numbers, set cudnn option
    utils.fix_seed(7)
    cudnn.enabled = True
    cudnn.benchmark = True

    # set model and optimized parameters
    device = torch.device('cuda:0')
    font_autoencoder = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device)
    font_autoencoder.load_state_dict(torch.load(MODEL_PATH))
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    emb_i = MLP.ReLU().to(device)
    emb_t = MLP.ReLU().to(device)
    models = [font_autoencoder, clip_model, emb_i, emb_t]
    
    # set optimizer, criterion, logger
    optimizer = torch.optim.Adam(list(emb_i.parameters())+list(emb_t.parameters()), lr=LR)
    criterion = losses.HMLC().to(device)
    
    # set dataloder, sampler
    train_img_paths, train_tag_paths = utils.LoadDatasetPaths("train")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    trainset = DMH_D(train_img_paths, train_tag_paths, IMG_HIERARCHY_PATH, TAG_HIERARCHY_PATH, tokenizer)
    sampler = DMH_BS(batch_size=BATCH_SIZE, drop_last=False, dataset=trainset)
    trainloader = torch.utils.data.DataLoader(trainset, sampler=sampler, num_workers=os.cpu_count(), batch_size=1, pin_memory=True)
    # to calcurate ARR
    valset = utils.EvalDataset(train_img_paths, train_tag_paths, tokenizer)
    train_evalloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    # for validation
    val_img_paths, val_tag_paths = utils.LoadDatasetPaths("val")
    valset = utils.EvalDataset(val_img_paths, val_tag_paths, tokenizer)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    # early stopping
    earlystopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, delta=0)

    # train and save results
    total_itr = 0
    meanARR_best = np.Inf
    for epoch in range(1, MAX_EPOCH + 1):
        print('Epoch {}/{}'.format(epoch, MAX_EPOCH))
        print('-'*10)

        # training and validation
        sampler.set_epoch(epoch)
        loss_img, loss_tag, loss, total_itr = train(trainloader, models, criterion, optimizer, total_itr, SAVE_FOLDER)
        ARR_img2tag_train, ARR_tag2img_train, meanARR_train = val(train_evalloader, models)
        ARR_img2tag_val, ARR_tag2img_val, meanARR_val = val(valloader, models)
        print(f"[train] loss_img: {loss_img:.4f}, loss_tag: {loss_tag:.4f}. ave: {loss:.4f}")
        print(f"[train] ARR_img2tag: {ARR_img2tag_train:.2f}, ARR_tag2img: {ARR_tag2img_train:.2f}. ave: {meanARR_train:.2f}")
        print(f"[val] ARR_img2tag: {ARR_img2tag_val:.2f}, ARR_tag2img: {ARR_tag2img_val:.2f}. ave: {meanARR_val:.2f}")
        
        # save results to wandb
        wandb.log({"loss_img_epoch": loss_img,
                "loss_tag_epoch": loss_tag, 
                "loss_epoch": loss,
                "ARR_img2tag_train": ARR_img2tag_train,
                "ARR_tag2img_train": ARR_tag2img_train,
                "meanARR_train": meanARR_train,
                "ARR_img2tag_val": ARR_img2tag_val,
                "ARR_tag2img_val": ARR_tag2img_val,
                "meanARR_val": meanARR_val,
                },
                step = total_itr
                )

        # save results to csv file
        f_epoch = open(f"{SAVE_FOLDER}/result_epoch.csv", 'a')
        csv_writer = csv.writer(f_epoch)
        csv_writer.writerow([epoch, loss_img, loss_tag, loss, ARR_img2tag_train, ARR_tag2img_train, meanARR_train, ARR_img2tag_val, ARR_tag2img_val, meanARR_val])
        f_epoch.close()
        
        # early stopping
        earlystopping(meanARR_val)
        if earlystopping.early_stop:
            print("Early stopping")
            break

        # save models and optimizer every 100 epochs
        if epoch%100==0 and epoch!=0:
            filename = SAVE_FOLDER+'/model/checkpoint_{:04d}.pth.tar'.format(epoch)
            save_contents = {'font_autoencoder':font_autoencoder.state_dict(), 'clip_model':clip_model.state_dict(),
                            'emb_i':emb_i.state_dict(), 'emb_t':emb_t.state_dict(), 'optimizer':optimizer.state_dict()}
            torch.save(save_contents, filename)
        
        # save models and optimizer if renew the best meanARR
        if meanARR_val<meanARR_best:
            filename = f"{SAVE_FOLDER}/model/best.pth.tar"
            save_contents = {'font_autoencoder':font_autoencoder.state_dict(), 'clip_model':clip_model.state_dict(),
                            'emb_i':emb_i.state_dict(), 'emb_t':emb_t.state_dict(), 'optimizer':optimizer.state_dict()}
            torch.save(save_contents, filename)
            meanARR_best = meanARR_val

    wandb.alert(
    title="WandBからの通知", 
    text=f"LR={LR}, BS={BATCH_SIZE}が終了しました．"
    )