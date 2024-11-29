'''
BCEロスで内積を +1 or -1にすることができるかの検証
'''
import torch
import numpy as np

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lib import utils
from models.HierarchicalDataset import HierarchicalDataset, HierarchicalBatchSampler, HierarchicalDatasetWithoutSampler
from models import MLP
from models import ExpMultiplier

# parameter from config
params = utils.get_parameters()
EXPT = params.expt
MAX_EPOCH = params.max_epoch
EARLY_STOPPING_PATIENCE = params.early_stopping_patience
LEARNING_RATE = params.learning_rate
BATCH_SIZE = 10
NUM_IMG_CLUSTERS = params.num_img_clusters
NUM_TAG_CLUSTERS = params.num_tag_clusters
TAG_PREPROCESS = params.tag_preprocess
TEMPERATURE = params.temperature
INITIAL_TEMPERATURE = 0.07
WEIGHTS = [1.0, 1.0, 1.0]

device = torch.device('cuda:0')
emb_img = MLP.ReLU().to(device)
emb_tag = MLP.ReLU().to(device)
temperature_class = getattr(ExpMultiplier, TEMPERATURE)
temperature = temperature_class(INITIAL_TEMPERATURE).to(device)

optimizer = torch.optim.Adam([
    {'params': emb_img.parameters(), 'lr': LEARNING_RATE[0]},
    {'params': emb_tag.parameters(), 'lr': LEARNING_RATE[1]},
    {'params': temperature.parameters(), 'lr': LEARNING_RATE[2]}
    ])

criterion_CE = torch.nn.CrossEntropyLoss().to(device)
criterion_BCE = torch.nn.BCEWithLogitsLoss().to(device)

trainset = HierarchicalDataset('train', EXPT, TAG_PREPROCESS, NUM_IMG_CLUSTERS, NUM_TAG_CLUSTERS)
sampler = HierarchicalBatchSampler(batch_size=BATCH_SIZE, dataset=trainset)
trainloader = torch.utils.data.DataLoader(trainset, sampler=sampler, shuffle=False, 
                                          num_workers=os.cpu_count(), batch_size=1, pin_memory=True)

# get features and labels
start = np.random.randint(1, 200, size=(1))[0]
start = 100
end = start+8
img_features = trainset.img_feature.to('cuda')[start:end]
tag_features = trainset.tag_feature.to('cuda')[start:end]
img_labels = torch.from_numpy(trainset.img_cluster).to('cuda')[start:end]
tag_labels = torch.from_numpy(trainset.tag_cluster).to('cuda')[start:end]

# prepare labels
pair_labels = torch.arange(img_features.shape[0]).to('cuda')
img_labels_transformed = (img_labels.unsqueeze(0)==img_labels.unsqueeze(1)).float()
tag_labels_transformed = (tag_labels.unsqueeze(0)==tag_labels.unsqueeze(1)).float()
labels = [pair_labels, img_labels_transformed, tag_labels_transformed]


_, tag_path = utils.load_dataset_paths('train')
for i in range(8):
    fontname = os.path.basename(tag_path[start+i])[:-4]
    tags = utils.get_font_tags(tag_path[start+i])
    print(f'{fontname}: {tags}')

for epoch in range(1, 5000 + 1):
    # forward
    with torch.set_grad_enabled(True):
        # get model outputs
        embedded_img_features = emb_img(img_features)
        embedded_tag_features = emb_tag(tag_features)

    pair_labels, img_labels, tag_labels = labels
    w_pair, w_img, w_tag = WEIGHTS

    # culuculate similarity matrix, logits
    similarity_matrix = torch.matmul(embedded_img_features, embedded_tag_features.T)
    # similarity_matrix_with_temperature = temperature(similarity_matrix)
    similarity_matrix_with_temperature = similarity_matrix
    logits_per_img = similarity_matrix_with_temperature
    logits_per_tag = similarity_matrix_with_temperature.T

    # culuculate loss_pair
    loss_pair_img = criterion_CE(logits_per_img, pair_labels)
    loss_pair_tag = criterion_CE(logits_per_tag, pair_labels)
    loss_pair = (loss_pair_img+loss_pair_tag)/2

    loss_img = criterion_BCE(logits_per_img, tag_labels)    # 画像から印象のロス
    loss_tag = criterion_BCE(logits_per_tag, img_labels)    # 印象から画像のロス

    # culuculate loss_total
    # loss_total = w_pair*loss_pair + w_img*loss_img + w_tag*loss_tag
    loss_total = 1*loss_pair+loss_tag+loss_img

    # backward and optimize
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

print(img_labels_transformed)
print(tag_labels_transformed)
print(similarity_matrix)
print(loss_pair.item(), loss_tag.item(), loss_img.item())