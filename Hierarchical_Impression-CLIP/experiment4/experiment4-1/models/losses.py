'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from __future__ import print_function

import torch
import torch.nn as nn


class SimCLR_loss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SimCLR_loss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features):
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
        features = features.view(features.shape[0], features.shape[1], -1)  # なくてもいい
        batch_size = features.shape[0]
        contrast_count = features.shape[1]
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        mask = mask.float().to(device)
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
        log_prob = logits-torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask*log_prob).sum(1) / mask.sum(1)    #正例の部分だけ抽出

        # loss
        # メモ: なんでここで(self.temperature/self.base_temperature)をかけてるんだろう？
        loss = - (self.temperature/self.base_temperature) * mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()   # loss.mesnと一緒

        return loss