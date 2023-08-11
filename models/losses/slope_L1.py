import torch.nn as nn

from up.utils.general.registry_factory import LOSSES_REGISTRY

__all__ = ['Slope_L1']

import torch
from torch.nn import functional as F
from up.utils.general.registry_factory import LOSSES_REGISTRY
from up.utils.general.log_helper import default_logger as logger


@LOSSES_REGISTRY.register('Slope_General')
class Slope_General(nn.Module):
    def __init__(self, loss_type='L1', loss_weight=1., reduction='mean',
                 positive_sampler=None):
        super(Slope_General, self).__init__()
        self.loss_weight = loss_weight
        func_mapping = {'L1': nn.L1Loss,
            'L2': nn.MSELoss,
            'SmoothL1': nn.SmoothL1Loss}
        self.loss = func_mapping[loss_type](reduction=reduction)
        self.positive_sampler = positive_sampler

    def gen_probe(self, preds):
        pass

    def forward(self, preds, target=None):
        if target is None:
            target = 1.
        n = len(preds)
        preds = torch.cat([i.view(1) for i in preds], dim=-1)
        loss = self.loss_weight * self.loss(preds, torch.tensor([target for _ in range(n)], device=preds.device))
        return loss


@LOSSES_REGISTRY.register('Slope_L1')
class Slope_L1(nn.Module):
    def __init__(self, loss_weight=1., reduction='mean'):
        super(Slope_L1, self).__init__()
        self.loss_weight = loss_weight
        self.loss = nn.L1Loss(reduction=reduction)

    def forward(self, preds, target=None):
        if target is None:
            target = 1.
        n = len(preds)
        preds = torch.cat([i.view(1) for i in preds], dim=-1)
        loss = self.loss_weight * self.loss(preds, torch.tensor([target for _ in range(n)], device=preds.device))
        return loss
