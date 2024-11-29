import torch


def calc_hierarchical_clip_loss(embedded_img_features, embedded_tag_features, 
                                temperature, weights, criterions, labels, loss_type, ce_bce, epoch):
    # 変数をほぐす
    criterion_CE, criterion_BCE = criterions
    pair_labels, img_labels, tag_labels = labels
    w_pair, w_img, w_tag = weights

    # culuculate similarity matrix, logits
    similarity_matrix = torch.matmul(embedded_img_features, embedded_tag_features.T)
    similarity_matrix_with_temperature = temperature(similarity_matrix)
    logits_per_img = similarity_matrix_with_temperature
    logits_per_tag = similarity_matrix_with_temperature.T
    
    # culuculate loss_pair
    loss_pair_img = criterion_CE(logits_per_img, pair_labels)
    loss_pair_tag = criterion_CE(logits_per_tag, pair_labels)
    loss_pair = (loss_pair_img+loss_pair_tag)/2

    # 学習法別の前処理
    if loss_type=='average':
        pass
    elif loss_type=='iterative':
        w_tag, w_img = (0, w_img) if epoch%2==1 else (w_tag, 0)
    # 画像➝印象と印象➝画像で同じクラスタの特徴だけ引き寄せるように学習
    elif loss_type=='label_and': 
        and_labels = img_labels*tag_labels
        img_labels = tag_labels = and_labels
    
    # culuculate loss_img, loss_tag
    if ce_bce=='CE':
        # CEの場合は，ラベルをソフトラベルに変換
        normalized_img_labels = img_labels/img_labels.sum(dim=1, keepdim=True)
        normalized_tag_labels = tag_labels/tag_labels.sum(dim=1, keepdim=True)
        loss_img = criterion_CE(logits_per_img, normalized_tag_labels)  # 画像から印象のロス
        loss_tag = criterion_CE(logits_per_tag, normalized_img_labels)  # 印象から画像のロス  
    elif ce_bce=='BCE':
        loss_img = criterion_BCE(logits_per_img, tag_labels)    # 画像から印象のロス
        loss_tag = criterion_BCE(logits_per_tag, img_labels)    # 印象から画像のロス
    
    # culuculate loss_total
    loss_total = w_pair*loss_pair + w_img*loss_img + w_tag*loss_tag
    loss_dict = {
        'total': loss_total, 
        'pair':  loss_pair, 
        'img':   loss_img, 
        'tag':   loss_tag
        }

    return loss_dict


def calc_hierarchical_clip_loss_eval(embedded_img_features, embedded_tag_features, 
                                     temperature, weights, criterions, labels, ce_bce):
    '''
    温度あり/なしそれぞれの場合, loss_total, loss_pair, loss_img, loss_tagを計算
    loss_totalは, loss_type=='average' として計算
    '''

    # 変数をほぐす
    criterion_CE, criterion_BCE = criterions
    pair_labels, img_labels, tag_labels = labels
    w_pair, w_img, w_tag = weights

    # culuculate similarity matrix
    similarity_matrix = torch.matmul(embedded_img_features, embedded_tag_features.T)
    similarity_matrix_with_temperature = temperature(similarity_matrix)

    def calc_losses(similarity_matrix):
        # get logits
        logits_per_img = similarity_matrix
        logits_per_tag = similarity_matrix.T
        
        # culuculate loss_pair
        loss_pair_img = criterion_CE(logits_per_img, pair_labels)
        loss_pair_tag = criterion_CE(logits_per_tag, pair_labels)
        loss_pair = (loss_pair_img+loss_pair_tag)/2
        
        # culuculate loss_img, loss_tag
        if ce_bce=='CE':
            # CEの場合は，ラベルをソフトラベルに変換
            normalized_img_labels = img_labels/img_labels.sum(dim=1, keepdim=True)
            normalized_tag_labels = tag_labels/tag_labels.sum(dim=1, keepdim=True)
            loss_img = criterion_CE(logits_per_img, normalized_tag_labels)  # 画像から印象のロス
            loss_tag = criterion_CE(logits_per_tag, normalized_img_labels)  # 印象から画像のロス  
        elif ce_bce=='BCE':
            loss_img = criterion_BCE(logits_per_img, tag_labels)    # 画像から印象のロス
            loss_tag = criterion_BCE(logits_per_tag, img_labels)    # 印象から画像のロス
        
        loss_total = w_pair*loss_pair + w_img*loss_img + w_tag*loss_tag
        loss_dict = {
            'total': loss_total.item(), 
            'pair':  loss_pair.item(), 
            'img':   loss_img.item(), 
            'tag':   loss_tag.item()
            }
        
        return loss_dict

    loss_without_temperature = calc_losses(similarity_matrix)
    loss_with_temperature = calc_losses(similarity_matrix_with_temperature)

    return loss_without_temperature, loss_with_temperature


# def calc_loss_pair(embedded_img_features, embedded_tag_features, temperature, criterion_CE):
#     # culuculate logits
#     logits_per_img = temperature(torch.matmul(embedded_img_features, embedded_tag_features.T))
#     logits_per_tag = logits_per_img.T 
#     # culuculate loss_pair
#     pair_labels = torch.arange(embedded_img_features.shape[0]).to('cuda')
#     loss_pair_img = criterion_CE(logits_per_img, pair_labels)
#     loss_pair_tag = criterion_CE(logits_per_tag, pair_labels)
#     loss_pair = (loss_pair_img+loss_pair_tag)/2
#     return loss_pair