# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Peiwen Lin, Xiangtai Li
# Email: linpeiwen@sensetime.com, lixiangtai@sensetime.com
import random

import torch.nn as nn
from torch.nn import functional as F

from up.utils.model.normalize import build_norm_layer
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.models.losses import build_loss
# from ..components import Aux_Module
import numpy as np
import torch
from ...models.losses.slope_L1 import Slope_L1
from ...models.backbones.dbb import DiverseBranchBlock

__all__ = ['fpn_decoder']


class Aux_Module(nn.Module):
    def __init__(self, in_planes, num_classes=19, normalize={'type': 'solo_bn'}, use_dbb=False):
        super(Aux_Module, self).__init__()

        self.aux = nn.Sequential(
            DiverseBranchBlock(in_planes, 256, kernel_size=3, stride=1, padding=1, normalize=normalize) if use_dbb else
            nn.Conv2d(in_planes, 256, kernel_size=3, stride=1, padding=1),

            nn.Identity() if use_dbb else
            build_norm_layer(256, normalize)[1],
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),

            DiverseBranchBlock(256, num_classes, kernel_size=1, stride=1, padding=0,
                normalize=normalize) if use_dbb else
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        res = self.aux(x)
        return res


class NormedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', k=1):
        super(NormedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias, padding_mode)
        self.k = k

    def forward(self, input):
        weight = self.weight
        norm = torch.norm(weight, p=2, dim=(1, 2, 3), keepdim=True)
        normalized_weight = weight / (norm.pow(self.k) + 1e-8)
        return F.conv2d(input, normalized_weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups)


@MODULE_ZOO_REGISTRY.register('LPCV_2023_Seg_Decoder')
class fpn_decoder(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, inplanes, with_aux=True, num_classes=19, inner_planes=512, head_planes=512,
                 normalize={'type': 'solo_bn'}, sizes=(1, 2, 3, 6), loss=None,
                 use_norm_conv=False,
                 norm_conv_k=1.,
                 use_slope_loss=False,
                 slope_loss_weight=1.,
                 use_dbb=False,
                 simple=False,
                 antialias=False,
                 upsample_mode='bilinear',
                 random_upsample=False,
                 aux_offset=1,
                 gt_long_type=False
                 ):
        super(fpn_decoder, self).__init__()
        # print(inplanes)
        self.prefix = self.__class__.__name__
        self.upsample_mode = upsample_mode
        self.upsample_mode_origin = upsample_mode
        self.aux_offset = aux_offset
        self.random_upsample = random_upsample
        self.gt_long_type = gt_long_type
        # self.ppm = PSPModule(inplanes, out_planes=inner_planes, normalize=normalize, sizes=sizes, use_dbb=use_dbb)
        if simple:
            self.head = nn.Conv2d(inplanes[0], num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.head = nn.Sequential(
                DiverseBranchBlock(inplanes[0], head_planes, kernel_size=3, padding=1,
                    dilation=1, normalize=normalize) if use_dbb else
                nn.Conv2d(inplanes[0], head_planes, kernel_size=3, padding=1, dilation=1, bias=False),

                nn.Identity() if use_dbb else
                build_norm_layer(head_planes, normalize)[1],

                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                NormedConv2d(head_planes, num_classes, kernel_size=1, stride=1, padding=0, bias=False,
                    k=norm_conv_k) if use_norm_conv else
                nn.Conv2d(head_planes, num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.tocaffe = False
        self.with_aux = with_aux
        self.antialias = antialias
        self.loss = build_loss(loss)
        if self.with_aux:
            self.aux_layer = Aux_Module(inplanes[self.aux_offset], num_classes, normalize, use_dbb=use_dbb)
        self.use_slope_loss = use_slope_loss
        if use_slope_loss:
            self.use_slope_loss = True
            self.slope_target = 1.
            self.slope_loss = Slope_L1(loss_weight=slope_loss_weight)

        self.aa_channels = 14
        filter_size = 3
        self.aa_pad = (filter_size - 1) // 2
        # weight = torch.ones(1, 1, filter_size, filter_size) / (filter_size * filter_size)
        aa_param = np.array([1., 2., 1.])
        aa_weight = torch.Tensor(aa_param[:, None] * aa_param[None, :])
        aa_weight = aa_weight / torch.sum(aa_weight)
        self.register_buffer('aa_weight', aa_weight[None, None, :, :].repeat((self.aa_channels, 1, 1, 1)))
        # self.aa_weight = weight.to('cuda')

    def forward(self, x):
        # x1, x2, x3, x4 = x['features']
        # size = x['size']
        # ppm_out = self.ppm(x4)
        pred = self.head(x['features'][0])
        if self.training and self.random_upsample:
            self.upsample_mode = random.choice(['bilinear', 'bicubic'])
        else:
            self.upsample_mode = self.upsample_mode_origin

        if not self.tocaffe:
            if self.antialias:
                n, c, h, w = pred.size()
                while h < 256:
                    pred = F.upsample(pred, size=(h * 2, w * 2), mode=self.upsample_mode, align_corners=True)
                    if self.training:
                        pred = F.conv2d(pred, weight=self.aa_weight, padding=self.aa_pad, groups=c)
                    n, c, h, w = pred.size()
                pred = F.upsample(pred, size=(512, 512), mode=self.upsample_mode, align_corners=True)
                if self.training:
                    pred = F.conv2d(pred, weight=self.aa_weight, padding=self.aa_pad, groups=c)
            else:
                pred = F.upsample(pred, size=(512, 512), mode=self.upsample_mode, align_corners=True)

        ret = dict()

        if self.training and self.use_slope_loss:
            slope_loss = self.slope_loss(x['slopes'], self.slope_target)
            ret.update({self.prefix + '.slope_loss': slope_loss})

            ret.update(
                {self.prefix + '.slope_value_min': torch.min(torch.tensor(x['slopes']).detach())})

            ret.update(
                {self.prefix + '.slope_value_mean': torch.mean(torch.tensor(x['slopes']).detach())})

            ret.update({self.prefix + '.slope_value_max': torch.max(torch.tensor(x['slopes']).detach())})

        if self.training and self.with_aux:
            gt_seg = x['gt_semantic_seg']
            aux_pred = self.aux_layer(x['features'][self.aux_offset])
            # aux_pred = F.upsample(aux_pred, size=(512, 512), mode='bilinear', align_corners=True)

            if self.antialias:
                n, c, h, w = aux_pred.size()
                while h < 256:
                    aux_pred = F.upsample(aux_pred, size=(h * 2, w * 2), mode=self.upsample_mode, align_corners=True)
                    aux_pred = F.conv2d(aux_pred, weight=self.aa_weight, padding=self.aa_pad, groups=c)
                    n, c, h, w = aux_pred.size()
                aux_pred = F.upsample(aux_pred, size=(512, 512), mode=self.upsample_mode, align_corners=True)

                aux_pred = F.conv2d(aux_pred, weight=self.aa_weight, padding=self.aa_pad, groups=c)
            else:
                aux_pred = F.upsample(aux_pred, size=(512, 512), mode=self.upsample_mode, align_corners=True)

            pred = pred, aux_pred
            loss = self.loss(pred, gt_seg)
            ret.update({f"{self.prefix}.loss": loss, "blob_pred": pred[0]})
        elif self.training:
            gt_seg = x['gt_semantic_seg']
            if self.gt_long_type:
                gt_seg = gt_seg.long()
            loss = self.loss(pred, gt_seg)
            ret.update({f"{self.prefix}.loss": loss, "blob_pred": pred[0]})
        else:
            ret.update({"blob_pred": pred})
        return ret
