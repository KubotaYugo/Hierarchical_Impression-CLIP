'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

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
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(HMLC, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.layer_penalty = torch.exp
        self.sup_con_loss = SupConLoss(temperature)

    def forward(self, features, labels):
        device = torch.device('cuda')
        mask = torch.ones(labels.shape).to(device)
        cumulative_loss = torch.tensor(0.0).to(device)
        max_loss_lower_layer = torch.tensor(float('-inf'))
        layer_loss_list = []
        for l in range(1,labels.shape[1]):
            mask[:, labels.shape[1]-l:] = 0
            layer_labels = labels * mask
            mask_labels = torch.stack([torch.all(torch.eq(layer_labels[i], layer_labels), dim=1)
                                       for i in range(layer_labels.shape[0])]).type(torch.uint8).to(device)
            layer_loss = self.sup_con_loss(features, mask=mask_labels)
            # hmce loss
            # メモ: maxとる対象が違う気がする → 実装の方に合わせた方が良いか？
            layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
            layer_loss_list.append(layer_loss.item())
            cumulative_loss += self.layer_penalty(torch.tensor(1/l).type(torch.float)) * layer_loss
            # renew max_loss_lower_layer
            max_loss_lower_layer = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
            # メモ: 下位レイヤで同じクラスのデータを1つだけに(こういう処理は論文には書かれていない)
            _, unique_indices = unique(layer_labels, dim=0)
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

    def forward(self, features, labels=None, mask=None):
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
        # メモ: ↓この1行いらない
        features = features.view(features.shape[0], features.shape[1], -1)
        mask = mask.float().to(device)
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
        logits_mask = torch.scatter(torch.ones_like(mask), 1, 
                                    torch.arange(batch_size*contrast_count).view(-1, 1).to(device), 0)
        mask = mask*logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits)*logits_mask  # 同じものどうし以外
        log_prob = logits-torch.log(exp_logits.sum(dim=1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask*log_prob).sum(1) / mask.sum(1)    #正例の部分だけ抽出

        # loss
        # メモ: なんでここで(self.temperature/self.base_temperature)をかけてるんだろう？
        loss = - (self.temperature/self.base_temperature) * mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()   # loss.mesnと一緒

        return loss