import torch

def calc_hierarchical_clip_loss(embedded_img_features, embedded_tag_features, 
                                temperature, weights, criterions, labels, loss_type, ce_bce, epoch):
    criterion_CE, criterion_BCE = criterions
    pair_labels, img_labels, tag_labels = labels
    w_pair, w_img, w_tag = weights

    # culuculate logits
    logits_per_img = temperature(torch.matmul(embedded_img_features, embedded_tag_features.T))
    logits_per_tag = logits_per_img.T
    # culuculate loss_pair
    loss_pair_img = criterion_CE(logits_per_img, pair_labels)
    loss_pair_tag = criterion_CE(logits_per_tag, pair_labels)
    loss_pair = (loss_pair_img+loss_pair_tag)/2
    # culuculate loss_img and loss_tag, total loss
    if ce_bce=='BCE':
        if loss_type=='average':
            loss_img = criterion_BCE(logits_per_img, tag_labels)    # 画像から印象のロス
            loss_tag = criterion_BCE(logits_per_tag, img_labels)    # 印象から画像のロス
            loss_total = w_pair*loss_pair + w_img*loss_img + w_tag*loss_tag
        elif loss_type=='iterative':
            # 画像➝印象➝画像の順で学習
            loss_img = criterion_BCE(logits_per_img, tag_labels)    # 画像から印象のロス
            loss_tag = criterion_BCE(logits_per_tag, img_labels)    # 印象から画像のロス
            if epoch%2==1:
                loss_total = w_pair*loss_pair + w_img*loss_img
            elif epoch%2==0:
                loss_total = w_pair*loss_pair + w_tag*loss_tag
        elif loss_type=='label_and':
            and_labels = img_labels*tag_labels   # 画像➝印象と印象➝画像で同じクラスタの特徴だけ引き寄せるように学習
            loss_img = criterion_BCE(logits_per_img, and_labels)
            loss_tag = criterion_BCE(logits_per_tag, and_labels)
            loss_total = w_pair*loss_pair + w_img*loss_img + w_tag*loss_tag
    elif ce_bce=='CE':
        if loss_type=='average':
            normalized_img_labels = img_labels/img_labels.sum(dim=1, keepdim=True)
            normalized_tag_labels = tag_labels/tag_labels.sum(dim=1, keepdim=True)
            loss_img = criterion_CE(logits_per_img, normalized_tag_labels)    # 画像から印象のロス
            loss_tag = criterion_CE(logits_per_tag, normalized_img_labels)    # 印象から画像のロス
            loss_total = w_pair*loss_pair + w_img*loss_img + w_tag*loss_tag
        elif loss_type=='iterative':
            # 画像➝印象➝画像の順で学習
            normalized_img_labels = img_labels/img_labels.sum(dim=1, keepdim=True)
            normalized_tag_labels = tag_labels/tag_labels.sum(dim=1, keepdim=True)
            loss_img = criterion_CE(logits_per_img, normalized_tag_labels)    # 画像から印象のロス
            loss_tag = criterion_CE(logits_per_tag, normalized_img_labels)    # 印象から画像のロス
            if epoch%2==1:
                loss_total = w_pair*loss_pair + w_img*loss_img
            elif epoch%2==0:
                loss_total = w_pair*loss_pair + w_tag*loss_tag
        elif loss_type=='label_and':
            and_labels = img_labels*tag_labels   # 画像➝印象と印象➝画像で同じクラスタの特徴だけ引き寄せるように学習
            normalized_and_labels = and_labels/and_labels.sum(dim=1, keepdim=True)
            loss_img = criterion_CE(logits_per_img, normalized_and_labels)
            loss_tag = criterion_CE(logits_per_tag, normalized_and_labels)
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