from transformers import CLIPTokenizer, CLIPModel
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
import csv
import wandb
import losses
from HierarchicalDataset import HierarchicalDataset, HierarchicalBatchSampler
import FontAutoencoder
import MLP

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils
from lib import eval_utils



class EarlyStopping:
    '''
    meanARRがpatience回以上更新されなければself.early_stopをTrueに
    '''
    def __init__(self, patience, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.early_stop = False
        self.min_value = np.Inf
    def __call__(self, value):
        if value >= self.min_value-self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            print(f'Validation meanARR decreased ({self.min_value} --> {value})')
            self.min_value = value


def train(dataloader, models, criterion1, criterion2, optimizer, total_itr, save_folder):
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
            loss_img, layer_loss_list_img = criterion1(features, img_labels)
            loss_tag, layer_loss_list_tag = criterion2(features, tag_labels)
            loss = (loss_img+loss_tag)/2
            loss_img_list.append(loss_img.item())
            loss_tag_list.append(loss_tag.item())
            loss_list.append(loss.item())
            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        # save results to wandb
        # save_dict = {'loss_itr':     loss.item(),
        #              'loss_img_itr': loss_img.item(),
        #              'loss_tag_itr': loss_tag.item()
        #             }
        # for i in range(len(layer_loss_list_img)):
        #     save_dict[f'loss_img_itr_layer{len(layer_loss_list_img)-i}'] = layer_loss_list_img[i]
        #     save_dict[f'loss_tag_itr_layer{len(layer_loss_list_img)-i}'] = layer_loss_list_tag[i]
        # wandb.log(save_dict, step=total_itr)
    
        # save results to csv file
        with open(f"{save_folder}/result_itr.csv", 'a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([total_itr, loss_img.item(), loss_tag.item(), loss.item()]+layer_loss_list_img+layer_loss_list_tag)

        total_itr += 1
    
    meanloss = np.mean(loss_list)
    loss_img = np.mean(loss_img_list)
    loss_tag = np.mean(loss_tag_list)
    loss_dict = {'mean':meanloss, 'img':loss_img, 'tag':loss_tag}

    return loss_dict, total_itr


def val(dataloader, models):
    # one epoch validation
    font_autoencoder, clip_model, emb_i, emb_t = models
    font_autoencoder.eval()
    clip_model.eval()
    emb_i.eval()
    emb_t.eval()

    with torch.no_grad():
        # extruct embedded features
        for idx, data in enumerate(dataloader):
            imgs, tokenized_tags = data
            imgs = imgs.cuda(non_blocking=True)
            tokenized_tags = tokenized_tags.cuda(non_blocking=True)

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
        
        # culculate Average Retrieval Rank
        ARR_tag2img = np.mean(eval_utils.retrieval_rank(similarity_matrix, "tag2img"))
        ARR_img2tag = np.mean(eval_utils.retrieval_rank(similarity_matrix, "img2tag"))
        meanARR = (ARR_img2tag+ARR_tag2img)/2
        ARR_dict = {'mean':meanARR, 'tag2img':ARR_tag2img, 'img2tag':ARR_img2tag}
        
    return ARR_dict 


if __name__ == '__main__':
    # parameter from config
    # config = wandb.config
    EXP = utils.EXP
    MAX_EPOCH = 100
    EARLY_STOPPING_PATIENCE = utils.EARLY_STOPPING_PATIENCE
    LOSS_TYPE = 'HMCE'
    NARROW_DOWN_INSTANCES = False
    LR = 1e-4
    BATCH_SIZE = 5

    MODEL_PATH = utils.MODEL_PATH
    IMG_CLUSTER_PATH = f'{EXP}/clustering/train/image_clusters.npz'
    TAG_CLUSTER_PATH = f'{EXP}/clustering/train/impression_clusters.npz'

    # make save directory
    SAVE_FOLDER = f'{EXP}/LR={LR}_BS={BATCH_SIZE}_{LOSS_TYPE}_{NARROW_DOWN_INSTANCES}/results'
    os.makedirs(f'{SAVE_FOLDER}/model', exist_ok=True)  

    # set run name and wandb dir
    # wandb.run.name = f'LR={LR}_BS={BATCH_SIZE}_{LOSS_TYPE}_{NARROW_DOWN_INSTANCES}'
    # wandb.init(dir=EXP)

    # fix random numbers, set cudnn option
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
    
    # set optimizer, criterion
    optimizer = torch.optim.Adam(list(emb_i.parameters())+list(emb_t.parameters()), lr=LR)
    criterion1 = losses.HMLC(loss_type=LOSS_TYPE, narrow_down_instances=NARROW_DOWN_INSTANCES).to(device)
    criterion2 = losses.HMLC(loss_type=LOSS_TYPE, narrow_down_instances=NARROW_DOWN_INSTANCES).to(device)

    # set dataloder, sampler for train
    train_img_paths, train_tag_paths = utils.load_dataset_paths("train")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    trainset = HierarchicalDataset(train_img_paths, train_tag_paths, IMG_CLUSTER_PATH, TAG_CLUSTER_PATH, tokenizer)
    sampler = HierarchicalBatchSampler(batch_size=BATCH_SIZE, drop_last=False, dataset=trainset)
    trainloader = torch.utils.data.DataLoader(trainset, sampler=sampler, num_workers=os.cpu_count(), batch_size=1, pin_memory=True)
    # to calcurate ARR
    train_evalset = utils.EvalDataset(train_img_paths, train_tag_paths, tokenizer)
    train_evalloader = torch.utils.data.DataLoader(train_evalset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
    # for validation
    val_img_paths, val_tag_paths = utils.load_dataset_paths("val")
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
        loss, total_itr = train(trainloader, models, criterion1, criterion2, optimizer, total_itr, SAVE_FOLDER)
        ARR_train = val(train_evalloader, models)
        ARR_val = val(valloader, models)
        print(f"[train] loss: {loss['mean']:.4f}, loss_img: {loss['img']:.4f}, loss_tag: {loss['tag']:.4f}")
        print(f"[train] meanARR: {ARR_train['mean']:.2f}, ARR_tag2img: {ARR_train['tag2img']:.2f}, ARR_img2tag: {ARR_train['img2tag']:.2f}")
        print(f"[val]   meanARR: {ARR_val['mean']:.2f}, ARR_tag2img: {ARR_val['tag2img']:.2f}, ARR_img2tag: {ARR_val['img2tag']:.2f}")
        
        # save results to wandb
        # wandb.log({"loss_epoch":        loss['mean'],
        #         "loss_img_epoch":    loss['img'],
        #         "loss_tag_epoch":    loss['tag'], 
        #         "meanARR_train":     ARR_train['mean'],
        #         "ARR_tag2img_train": ARR_train['tag2img'],
        #         "ARR_img2tag_train": ARR_train['img2tag'],
        #         "meanARR_val":       ARR_val['mean'],
        #         "ARR_tag2img_val":   ARR_val['tag2img'],
        #         "ARR_img2tag_val":   ARR_val['img2tag'],
        #         },
        #         step = total_itr
        #         )

        # save results to csv file
        with open(f"{SAVE_FOLDER}/result_epoch.csv", 'a') as f_epoch:
            csv_writer = csv.writer(f_epoch)
            csv_writer.writerow([epoch,
                                loss['mean'], loss['img'], loss['tag'], 
                                ARR_train['mean'], ARR_train['tag2img'], ARR_train['img2tag'], 
                                ARR_val['mean'],   ARR_val['tag2img'],   ARR_val['img2tag']])
        
        # early stopping
        earlystopping(ARR_val['mean'])
        if earlystopping.early_stop:
            print("Early stopping")
            break

        # save models and optimizer every 100 epochs
        if epoch%100==0 and epoch!=0:
            filename = f'{SAVE_FOLDER}/modelcheckpoint_{epoch:04d}.pth.tar'
            save_contents = {'font_autoencoder':font_autoencoder.state_dict(), 'clip_model':clip_model.state_dict(),
                            'emb_i':emb_i.state_dict(), 'emb_t':emb_t.state_dict(), 'optimizer':optimizer.state_dict()}
            torch.save(save_contents, filename)
        
        # save models and optimizer if renew the best meanARR
        if ARR_val['mean']<meanARR_best:
            filename = f"{SAVE_FOLDER}/model/best.pth.tar"
            save_contents = {'font_autoencoder':font_autoencoder.state_dict(), 'clip_model':clip_model.state_dict(),
                            'emb_i':emb_i.state_dict(), 'emb_t':emb_t.state_dict(), 'optimizer':optimizer.state_dict()}
            torch.save(save_contents, filename)
            meanARR_best = ARR_val['mean']

    # wandb.alert(
    # title="WandBからの通知", 
    # text=f"LR={LR}, BS={BATCH_SIZE}が終了しました．"
    # )