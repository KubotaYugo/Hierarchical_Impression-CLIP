import torch
import torch.nn.functional as F
import numpy as np


def sup_con_loss(logits, label):
    # for numerical stability
    logits_max = logits.max(dim=1, keepdim=True)[0]
    logits = logits - logits_max.detach()
    # compute log_prob
    exp_logits = torch.exp(logits) 
    sum_exp_logits = exp_logits.sum(dim=1, keepdim=True)
    loss_pos_neg = logits - torch.log(sum_exp_logits)
    loss_pos = loss_pos_neg*label
    # 1つのインスタンスに対する正例について平均
    loss_instance = -1 * loss_pos.sum(dim=1) / label.sum(dim=1)
    del logits_max, logits, exp_logits, sum_exp_logits, loss_pos_neg, loss_pos, label
    return loss_instance

def hierarchical_sup_con_loss(logit, clusterID, loss_type, direction):
    layer_loss_dict = {}
    for i in range(len(clusterID[0])):
        # print(f'階層: {i}')
        # ラベルの前処理
        clusterID_prune = np.array([s[:i + 1] for s in clusterID])
        labels = (clusterID_prune[:, None] == clusterID_prune[None, :]).astype(np.uint8)
        labels = torch.from_numpy(labels).to('cuda')

        # ロスの計算
        if loss_type == 'SupCon':
            loss_temp = sup_con_loss(logit, labels)
        elif loss_type == 'BCE':
            # メモリが足りない
            labels = labels.to(torch.float64)
            loss_temp = F.binary_cross_entropy(torch.sigmoid(logit), labels, reduction='none')
            loss_temp = loss_temp.mean(dim=1)
        layer_loss_dict[f'layer_loss_{direction}_{i+1}'] = loss_temp.mean().to('cpu').detach().item()
        loss = torch.max(loss, loss_temp) if i!=0 else loss_temp    # 階層間のロスの比較
        del clusterID_prune, labels, loss_temp
    return loss.mean(), layer_loss_dict

def calc_hierarchical_clip_loss(embedded_img_features, embedded_tag_features, 
                                temperature, weights, img_clusterID, tag_clusterID, loss_type):
    
    # 変数をほぐす
    w_img2tag, w_tag2img = weights

    # culuculate similarity matrix, logits
    similarity_matrix = torch.matmul(embedded_img_features, embedded_tag_features.T)
    similarity_matrix_with_temperature = temperature(similarity_matrix)
    logits_per_img = similarity_matrix_with_temperature
    logits_per_tag = similarity_matrix_with_temperature.T

    # culuculate losses
    loss_img2tag, layer_loss_dict_img2tag = hierarchical_sup_con_loss(logits_per_img, tag_clusterID, loss_type, 'img2tag')
    loss_tag2img, layer_loss_dict_tag2img = hierarchical_sup_con_loss(logits_per_tag, img_clusterID, loss_type, 'tag2img')
    loss_total = w_img2tag*loss_img2tag + w_tag2img*loss_tag2img
    loss_dict = {
        'total':    loss_total,  
        'img2tag':  loss_img2tag, 
        'tag2img':  loss_tag2img
        }

    return loss_dict, layer_loss_dict_img2tag, layer_loss_dict_tag2img


def calc_loss_eval(embedded_img_features, embedded_tag_features, temperature):
    '''
    温度あり/なしそれぞれの場合のloss_pairを計算
    '''

    def calc_loss_pair(similarity_matrix):
        # get logits
        logits_per_img = similarity_matrix
        logits_per_tag = similarity_matrix.T
        
        # culuculate loss_pair
        criterion_CE = torch.nn.CrossEntropyLoss()
        pair_labels = torch.arange(logits_per_img.shape[0]).to('cuda')
        loss_pair_img2tag = criterion_CE(logits_per_img, pair_labels)
        loss_pair_tag2img = criterion_CE(logits_per_tag, pair_labels)
        loss_pair = (loss_pair_img2tag+loss_pair_tag2img)/2

        loss_dict = {
            'pair':         loss_pair.item(), 
            'pair_img2tag': loss_pair_img2tag.item(), 
            'pair_tag2img': loss_pair_tag2img.item()
            }
        
        return loss_dict

    # culuculate similarity matrix
    similarity_matrix = torch.matmul(embedded_img_features, embedded_tag_features.T)
    similarity_matrix_with_temperature = temperature(similarity_matrix)
    loss_without_temperature = calc_loss_pair(similarity_matrix)
    loss_with_temperature = calc_loss_pair(similarity_matrix_with_temperature)

    return loss_without_temperature, loss_with_temperature