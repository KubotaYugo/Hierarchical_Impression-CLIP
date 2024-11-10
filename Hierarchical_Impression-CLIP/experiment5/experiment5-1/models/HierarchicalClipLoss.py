import torch

def calc_hierarchical_clip_loss(embedded_img_features, embedded_tag_features, temperature, weights, criterions, labels):
    criterion_CE, criterion_BCE = criterions
    pair_labels, img_labels, tag_labels = labels
    w_pair, w_img, w_tag = weights

    # culuculate logits
    logits_per_img = temperature(torch.matmul(embedded_img_features, embedded_tag_features.T))
    # print(tag_labels)
    # print(img_labels)
    # print(torch.matmul(embedded_img_features, embedded_tag_features.T))
    logits_per_tag = logits_per_img.T
    # culuculate loss_pair
    loss_pair_img = criterion_CE(logits_per_img, pair_labels)
    loss_pair_tag = criterion_CE(logits_per_tag, pair_labels)
    loss_pair = (loss_pair_img+loss_pair_tag)/2
    # culuculate loss_img and loss_tag
    loss_img = criterion_BCE(logits_per_img, tag_labels)    # 画像から印象のロス
    loss_tag = criterion_BCE(logits_per_tag, img_labels)    # 印象から画像のロス
    # total loss
    loss_total = w_pair*loss_pair + w_img*loss_img + w_tag*loss_tag

    return loss_total, loss_pair, loss_img, loss_tag

def calc_loss_pair(embedded_img_features, embedded_tag_features, temperature, criterion_CE):
    # culuculate logits
    logits_per_img = temperature(torch.matmul(embedded_img_features, embedded_tag_features.T))
    logits_per_tag = logits_per_img.T 
    # culuculate loss_pair
    pair_labels = torch.arange(embedded_img_features.shape[0]).to('cuda')
    loss_pair_img = criterion_CE(logits_per_img, pair_labels)
    loss_pair_tag = criterion_CE(logits_per_tag, pair_labels)
    loss_pair = (loss_pair_img+loss_pair_tag)/2

    return loss_pair