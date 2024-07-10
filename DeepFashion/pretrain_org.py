"""
pretrain.pyをわかりやすく書き直したもの
分散処理のところを削除
"""

import os
import sys
import argparse
from hierarchical_dataset_modified import DeepFashionHierarchihcalDataset, HierarchicalBatchSampler
from util import adjust_learning_rate, warmup_learning_rate, TwoCropTransform
from losses import HMLC
import resnet_modified
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import time
import shutil
import math
import numpy as np
import random
import csv


def parse_option():
    parser = argparse.ArgumentParser(description='Training/finetuning on Deep Fashion Dataset')
    parser.add_argument('--data', default='', type=str, 
                        help='path to dataset, the superset of train/val')
    parser.add_argument('--save_freq', default=20, type=int, 
                        help='save frequency')
    parser.add_argument('--model', default='resnet50', type=str)
    parser.add_argument('--train-listfile', default='', type=str,
                        help='training file with annotation')
    parser.add_argument('--val-listfile', default='', type=str,
                        help='validation file with annotation')
    parser.add_argument('--class-map-file', default='', type=str,
                        help='class mapping between str and int')
    parser.add_argument('--repeating-product-file', default='', type=str,
                        help='repeating product ids file')
    parser.add_argument('--mode', default='train', type=str,
                        help='Train or val')
    parser.add_argument('--input-size', default=224, type=int,
                        help='input size')
    parser.add_argument('--num-classes', default=17, type=int,
                        help='number of classes')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=512, type=int,
                        help='mini-batch size (default: 512)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--pretrained', 
                        action='store_false',
                        help='use pre-trained model')
    parser.add_argument('--feature-extract', 
                        action='store_false',
                        help='When flase, finetune the whole model; else only update the reshaped layer para')
    parser.add_argument('--save-root-folder', default='', type=str,
                        help='root folder for save models and tensorboard files')
    # temperature
    parser.add_argument('--temp', default=0.07, type=float,
                        help='temperature for loss function')
    # optimization
    parser.add_argument('--learning_rate', default=0.01, type=float,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', default='30,60,90', type=str,
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', default=0.1, type=float,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    #other setting
    parser.add_argument('--cosine', 
                        action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', 
                        action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--loss', default='hmce', type=str,
                        help='loss type', choices=['hmc', 'hce', 'hmce'])
    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    # warm-up for large-batch training,
    if args.batch_size >= 256:
        args.warm = True
    if args.warm:
        args.model_name = '{}_warm'.format(args.model)
        args.warmup_from = 0.01
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * \
                             (1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate
    return args


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm


def set_resnet50_pretrain(name):
    model = resnet_modified.MyResNet(name=name)
    state_dict = torch.load(args.ckpt, map_location='cpu')
    model_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        k = "encoder."+k
        new_state_dict[k] = v
    new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}   # delete unnecessary keys
    state_dict = new_state_dict
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model

def set_parameter_requires_grad(model, feature_extracting):
    # Select which params to finetune
    if feature_extracting:
        for name, param in model.named_parameters():
            if name.startswith('encoder.layer4'):
                param.requires_grad = True
            elif name.startswith('encoder.layer3'):
                param.requires_grad = True
            elif name.startswith('head'):
                param.requires_grad = True
            else:
                param.requires_grad = False


def setup_optimizer(model_ft, lr, momentum, weight_decay, feature_extract):
    # set features to update on optimizer
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
    optimizer_ft = torch.optim.SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer_ft


def load_deep_fashion_hierarchical(args):
    if args.mode=="train":
        data_list_file = args.train_listfile
        transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=args.input_size, scale=(0.8, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4)], p=0.8),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], 
                                         [0.229, 0.224, 0.225]),])
    elif args.mode=="val" or args.mode=="test":
        data_list_file = args.val_listfile
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], 
                                         [0.229, 0.224, 0.225]),])

    dataset = DeepFashionHierarchihcalDataset(os.path.join(args.data, data_list_file),
                                              os.path.join(args.data, args.class_map_file),
                                              os.path.join(args.data, args.repeating_product_file),
                                              transform=TwoCropTransform(transform))
    sampler = HierarchicalBatchSampler(batch_size=args.batch_size, drop_last=False, dataset=dataset)
    # メモ: なんでbatch_size=1なんだろう？
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, num_workers=os.cpu_count(), batch_size=1, pin_memory=True)
    return dataloader, sampler


def train(dataloader, model, criterion, optimizer, epoch, args, logger):
    # one epoch training
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    end = time.time()

    # Each epoch has a training and/or validation phase
    progress = ProgressMeter(len(dataloader), [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))
    model.train()  # Set model to training mode

    # Iterate over data.
    for idx, (images, labels) in enumerate(dataloader):
        data_time.update(time.time() - end)
        labels = labels.squeeze()
        images = torch.cat([images[0].squeeze(), images[1].squeeze()], dim=0)
        images = images.cuda(non_blocking=True)
        labels = labels.squeeze().cuda(non_blocking=True)
        bsz = labels.shape[0] #batch size
        warmup_learning_rate(args, epoch, idx, len(dataloader), optimizer)

        # forward
        with torch.set_grad_enabled(True):
            # Get model outputs and calculate loss
            features = model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(features, labels)
            losses.update(loss.item(), bsz)
            # backward + optimize only if in training phase
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        sys.stdout.flush()
        if idx % args.print_freq == 0:
            progress.display(idx)
        
        # save results
        logger.add_scalar('loss', loss.item(), len(dataloader)*(epoch-1)+idx)
        f = open(f"{args.save_root_folder}/result.csv", 'a')
        csv_writer = csv.writer(f)
        csv_writer.writerow([epoch, len(dataloader)*(epoch-1)+idx, loss.item()])
        f.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    # メモ: この関数は機能していない
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    # get args, fix random numbers, set cudnn option
    args = parse_option()
    fix_seed(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True

    # make directries to save
    args.save_folder = args.save_root_folder + '/model'
    args.tb_folder = args.save_root_folder + '/tensorboard'
    os.makedirs(args.save_folder, exist_ok=True)
    os.makedirs(args.tb_folder, exist_ok=True)

    # set model and optimized parameters
    device = torch.device('cuda:0')
    model = set_resnet50_pretrain(name='resnet50').to(device)
    set_parameter_requires_grad(model, args.feature_extract)
    
    # set optimizer, ceiterion, dataloder, sampler, logger
    optimizer = setup_optimizer(model, args.learning_rate, args.momentum, args.weight_decay, args.feature_extract)
    criterion = HMLC(temperature=args.temp, loss_type=args.loss, layer_penalty=torch.exp).to(device)
    dataloader, sampler = load_deep_fashion_hierarchical(args=args)
    logger = SummaryWriter(args.tb_folder)
    
    # train and save results
    for epoch in range(1, args.epochs + 1):
        print('Epoch {}/{}'.format(epoch, args.epochs + 1))
        print('-'*10)

        adjust_learning_rate(args, optimizer, epoch)
        train(dataloader, model, criterion, optimizer, epoch, args, logger)
        
        output_file = args.save_folder+'/checkpoint_{:04d}.pth.tar'.format(epoch)
        save_contents = {'epoch': epoch+1, 'arch': args.model, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(save_contents, is_best=False, filename=output_file)