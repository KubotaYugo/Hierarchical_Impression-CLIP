from __future__ import print_function

import torch
import torch.nn as nn


def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)


class HMLC(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, loss_type='HMCE', narrow_down_instances=True):
        super(HMLC, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.loss_type = loss_type
        self.narrow_down_instances = narrow_down_instances
        self.layer_penalty = torch.exp
        self.sup_con_loss = SupConLoss(temperature)

    def forward(self, features, labels):
        device = torch.device('cuda')
        mask = torch.ones(labels.shape).to(device)
        cumulative_loss = torch.tensor(0.0).to(device)
        max_loss_lower_layer = torch.tensor(float('-inf')).to(device)
        layer_loss_list = []
        for layer in range(labels.shape[1]):
            # ラベルの処理
            mask[:, labels.shape[1]-layer:] = 0
            layer_labels = labels * mask
            mask_labels = torch.stack([torch.all(torch.eq(layer_labels[i], layer_labels), dim=1)
                                       for i in range(layer_labels.shape[0])]).type(torch.uint8).to(device)
            # ロス計算
            layer_loss_wo_penalty, pos_pair_loss_max = self.sup_con_loss(features, mask_labels, max_loss_lower_layer)
            layer_penalty = self.layer_penalty(torch.tensor(1/(layer+1)).type(torch.float))      
            if self.loss_type=='HMCE':       
                layer_loss = layer_penalty * layer_loss_wo_penalty
                max_loss_lower_layer = torch.max(max_loss_lower_layer, pos_pair_loss_max)
            elif self.loss_type=='HMC': 
                layer_loss = layer_penalty * layer_loss_wo_penalty
            elif self.loss_type=='HCE':        
                layer_loss = layer_loss_wo_penalty  # メモ: ここミスってた(= layer_penaltyにしてた)
                max_loss_lower_layer = torch.max(max_loss_lower_layer, pos_pair_loss_max)
            cumulative_loss += layer_loss
            layer_loss_list.append(layer_loss.item())

            # メモ: 下位レイヤで同じクラスタのデータを1つだけに(元論文には書かれていないが実装してあった)
            if self.narrow_down_instances and layer!=labels.shape[1]-1:  # メモ: 階層を深くするときはここの挙動を要確認
                label_mask = torch.ones(labels.shape).to(device)
                label_mask[:, labels.shape[1]-layer-1:] = 0
                _, unique_indices = unique(labels*label_mask, dim=0)
                labels = labels[unique_indices]
                mask = mask[unique_indices]
                features = features[unique_indices]

        return cumulative_loss/labels.shape[1], layer_loss_list


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, mask=None, max_loss_lower_layer=torch.tensor(float('-inf'))):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device('cuda')
        mask = mask.float().to(device)
        max_loss_lower_layer = max_loss_lower_layer.to(device)
        batch_size = features.shape[0]
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
            
        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(contrast_feature, contrast_feature.T), self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(contrast_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size*contrast_count).view(-1, 1).to(device), 0)
        mask = mask*logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits)*logits_mask  # 同じものどうし以外
        log_prob = logits-torch.log(exp_logits.sum(dim=1, keepdim=True))    # 分子マイナス分母
        loss_pos_neg = - (self.temperature/self.base_temperature) * log_prob
        
        # max_loss_lower_layerより小さいところをmax_loss_lower_layerに
        loss_pos_neg_replaced = torch.where(loss_pos_neg<max_loss_lower_layer, max_loss_lower_layer, loss_pos_neg)
        # 正例のところだけに
        pos_pair_loss = loss_pos_neg_replaced*mask
        # 1つのインスタンスに対する正例について平均
        instance_loss = pos_pair_loss.sum(dim=1)/mask.sum(dim=1)  
        # バッチ内で平均 (先行研究の論文では，ここは総和になっている)
        batch_loss = instance_loss.mean()

        # 正例のSimCLRロスの最大値
        pos_pair_loss_max = torch.max(loss_pos_neg[mask==1])
        
        return batch_loss, pos_pair_loss_max