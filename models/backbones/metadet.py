import math
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Sequence

from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.utils.model.initializer import initialize_from_cfg
from up.utils.general.log_helper import default_logger as logger
# from .ac_kernels.ac import D3L_ACBlock
from .switchable_activations.utils_ds import Learnable_Relu
from .dbb import DiverseBranchBlock
from up.utils.model.normalize import build_norm_layer
import math


def _scale_filters(filters, multiplier=1.0, base=8):
    """Scale the filters accordingly to (multiplier, base)."""
    round_half_up = int(int(filters) * multiplier / base + 0.5)
    result = int(round_half_up * base)
    return max(result, base)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()

    def forward(self, x):
        x = torch.clamp(x + 3, min=0, max=6)
        return x / 6


class AvgPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean([2, 3], keepdim=True)


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        # norm_layer: Optional[Callable[..., nn.Module]] = None,
        normalize={'type': 'solo_bn'},
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1, block_type='AC', dbb_mid_expand_factor=1.,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if normalize is None:
            normalize = {'type': 'solo_bn'}
        if activation_layer is None:
            activation_layer = nn.ReLU
        if kernel_size == 3 and groups == 1:
            if block_type == 'AC':
                super().__init__(
                    D3L_ACBlock(in_planes=in_planes, out_planes=out_planes, kernel_size=kernel_size, stride=stride,
                        padding=padding, bias=False),
                    activation_layer(inplace=True))
            elif block_type == 'DBB':
                super().__init__(
                    DiverseBranchBlock(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                        stride=stride, padding=padding, groups=groups,
                        internal_channels_1x1_3x3=dbb_mid_expand_factor * out_planes,
                        normalize=normalize),
                    activation_layer(inplace=True))
            else:
                raise NotImplementedError
        else:
            if block_type == 'DBB':
                super().__init__(
                    DiverseBranchBlock(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                        stride=stride, padding=padding, groups=groups,
                        internal_channels_1x1_3x3=dbb_mid_expand_factor * out_planes,
                        normalize=normalize),
                    activation_layer(inplace=True))
            else:
                super().__init__(
                    nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                        bias=False),
                    # norm_layer(out_planes),
                    build_norm_layer(out_planes, normalize)[1],
                    activation_layer(inplace=True)
                )
        self.out_channels = out_planes


class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _scale_filters(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.h_sigmoid = h_sigmoid()
        # self.pool = AvgPool()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def _scale(self, input: Tensor, inplace: bool) -> Tensor:
        scale = self.pool(input)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return self.h_sigmoid(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input, True)
        return scale * input


class InvertedResidualConfig:

    def __init__(self, input_channels: int, kernel: int, e_ratio: float, out_channels: int, stride: int,
                 s_ratio: float = 1.0, use_se: bool = False, activation: str = 'RE', dilation: int = 1,
                 use_learnable_relu=False, width_mult: float = 1.0, more_act=False, more_act2=False, base=8,
                 force_connect=False, dense_connect=False, no_connect=False):
        self.kernel = kernel
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.s_channels = self.adjust_channels(self.input_channels, s_ratio, base=base)
        self.expanded_channels = self.adjust_channels(self.out_channels, e_ratio, base=base)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.use_learnable_relu = use_learnable_relu
        self.stride = stride
        self.dilation = dilation
        self.more_act = more_act
        self.more_act2 = more_act2
        self.force_connect = force_connect
        self.dense_connect = dense_connect
        self.no_connect = no_connect

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, base: int = 8):
        return _scale_filters(channels, width_mult, base)


class InvertedResidual(nn.Module):

    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module],
                 se_layer: Callable[..., nn.Module] = SqueezeExcitation):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = Learnable_Relu if cnf.use_learnable_relu else nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(ConvBNActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=1,
                norm_layer=norm_layer, activation_layer=activation_layer))

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
            stride=stride, dilation=cnf.dilation, groups=cnf.expanded_channels,
            norm_layer=norm_layer, activation_layer=activation_layer))
        if cnf.use_se:
            layers.append(se_layer(cnf.expanded_channels))

        # project
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer,
            activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class FusedConv(nn.Module):

    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module],
                 se_layer: Callable[..., nn.Module] = SqueezeExcitation):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = Learnable_Relu if cnf.use_learnable_relu else nn.Hardswish if cnf.use_hs else nn.ReLU

        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(ConvBNActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
            stride=stride, dilation=cnf.dilation, groups=1,
            norm_layer=norm_layer, activation_layer=activation_layer))

        if cnf.use_se:
            layers.append(se_layer(cnf.expanded_channels))

        # project
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer,
            activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class FusedConv_DBB_single(nn.Module):

    def __init__(self, cnf: InvertedResidualConfig,  # norm_layer: Callable[..., nn.Module],
                 normalize={'type': 'solo_bn'},
                 se_layer: Callable[..., nn.Module] = SqueezeExcitation, dbb_mid_expand_factor=2.):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')
        stride = 1 if cnf.dilation > 1 else cnf.stride
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        self.conn = nn.Identity()
        self.dense_connect = cnf.dense_connect
        self.force_connect = cnf.force_connect
        self.no_connect = cnf.no_connect
        self.is_downsample_block = False
        if self.no_connect:
            self.use_res_connect = False
        elif cnf.force_connect:
            self.use_res_connect = True
            if not (cnf.stride == 1 and cnf.input_channels == cnf.out_channels):
                self.conn = ConvBNActivation(cnf.input_channels, cnf.out_channels, kernel_size=1,
                    stride=stride, dilation=cnf.dilation, groups=1,
                    # norm_layer=norm_layer,
                    normalize=normalize,
                    activation_layer=nn.Identity, block_type='DBB',
                    dbb_mid_expand_factor=2 * dbb_mid_expand_factor)
            else:
                self.is_downsample_block = True

        # layers: List[nn.Module] = []
        activation_layer = Learnable_Relu if cnf.use_learnable_relu else nn.Hardswish if cnf.use_hs else nn.ReLU
        self.conv1 = ConvBNActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
            stride=stride, dilation=cnf.dilation, groups=1,
            # norm_layer=norm_layer,
            normalize=normalize,
            activation_layer=nn.Identity, block_type='DBB',
            dbb_mid_expand_factor=dbb_mid_expand_factor)
        # self.block = nn.Sequential(*layers)

        # self.post_act = nn.Identity()
        # self.more_act = cnf.more_act
        # if self.more_act or self.dense_connect:
        #     self.post_act = activation_layer()

        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1
        self.act = activation_layer()

        self.forward = self.forward_simple
        self.fuse_model = self.switch_to_deploy

    def switch_to_deploy(self):
        if self.dense_connect:
            self.conv1[0].switch_to_deploy()
            c_in, c_out, k_w, k_h = self.conv1[0].dbb_reparam.weight.data.shape
            if not self.is_downsample_block and c_in == c_out and k_w == k_h == 3 and self.forward == self.forward_simple \
                and isinstance(self.conn, nn.Identity) and self.use_res_connect:
                for i in range(c_in):
                    self.conv1[0].dbb_reparam.weight.data[i][i][1][1] += 1
                self.forward = self.forward_dense_connect_merge_all

    def forward_simple(self, input: Tensor) -> Tensor:
        result = self.conv1(input)
        if self.use_res_connect:
            result += self.conn(input)
        result = self.act(result)
        return result

    def forward_dense_connect_merge_all(self, input: Tensor) -> Tensor:
        result = self.act(self.conv1(input))
        return result


class MBDet_DepConv3x3_Magic_single(nn.Module):

    def __init__(
        self,
        out_layers, out_strides, frozen_layers,
        initializer=None, normalize={'type': 'solo_bn'},
        in_channel=1, use_maxpool=True,
        more_act=False,
        width=[[1, 8], [8, 16], [16, 24, 24], [24, 32, 32, 32, 32, 32], [32, 32, 32, 32], [32, 40, 40]],
        expand_ratio=[[1], [1], [1, 1], [1, 1, 1, 1, 1], [1, 1, 1], [1, 1]],
        force_connect=False,
        dbb=False, dbb_mid_expand_factor=2., dense_connect=False,
        no_connect=False, simple_stem=False, ghost=False,
        remove_first_stride=False, dropout_prob=0.,
        input_resize=False, input_resize_ratio=1., input_resize_mode='area',input_resize_align_corners=None,
        out_planes=None
    ) -> None:
        super().__init__()

        self.input_channel = in_channel
        if in_channel != 1:
            logger.info("Backbone mode is BGR, input channle is 3")
        else:
            logger.info("Backbone mode is GRAY, input channle is 1")
        self.tocaffe = False
        self.out_layers = out_layers
        self.out_strides = out_strides
        self.frozen_layers = frozen_layers
        assert len(out_layers) == len(out_strides)
        self.use_maxpool = use_maxpool
        self.input_resize = input_resize
        self.input_resize_ratio = input_resize_ratio
        self.input_resize_align_corners = input_resize_align_corners
        self.input_resize_mode=input_resize_mode
        if out_planes is None:

            if simple_stem and not remove_first_stride:
                layer_out_planes = [w[-1] for w in width]
            else:
                layer_out_planes = [w[-1] for w in width[1:]]
            self.out_planes = [layer_out_planes[i] for i in out_layers]
        else:
            self.out_planes = out_planes
        # assert min(out_layers) >= 0 and max(out_layers) <= 4, out_layers

        bneck_conf = InvertedResidualConfig
        if simple_stem:
            raise NotImplementedError
        else:
            seq = []
        for i, channel in enumerate(width):
            if not (remove_first_stride and i == 0):
                seq.append([])
            for j in range(len(channel) - 1):
                if dbb:
                    seq[-1].append(FusedConv_DBB_single(
                        bneck_conf(channel[j], 3, expand_ratio[i][j], channel[j + 1], stride=2 if j == 0 else 1,
                            more_act=more_act, force_connect=force_connect, dense_connect=dense_connect,
                            no_connect=no_connect),
                        normalize=normalize,
                        dbb_mid_expand_factor=dbb_mid_expand_factor,
                    ))
                else:
                    raise NotImplementedError
            if dropout_prob:
                seq[-1].append(nn.Dropout(p=dropout_prob))
        self.input_layer = nn.Sequential(*seq[0])
        self.layer0 = nn.Sequential(*seq[1])
        self.layer1 = nn.Sequential(*seq[2])
        self.layer2 = nn.Sequential(*seq[3])
        if max(self.out_layers) >= 3:
            self.layer3 = nn.Sequential(*seq[4])
        if max(self.out_layers) >= 4:
            self.layer4 = nn.Sequential(*seq[5])
        if max(self.out_layers) >= 5:
            self.layer4 = nn.Sequential(*seq[6])
        initialize_from_cfg(self, initializer)

    def forward(self, x):
        x = x['image']
        if self.use_maxpool:
            x = self.pool(x)
        if self.input_resize and not self.tocaffe:
            x = F.interpolate(x, scale_factor=self.input_resize_ratio, mode=self.input_resize_mode ,
                align_corners=self.input_resize_align_corners)
        x = self.input_layer(x)
        c1 = self.layer0(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        if -1 in self.out_layers:
            outs = [x, c1, c2, c3]
        else:
            outs = [c1, c2, c3]
        if max(self.out_layers) >= 3:
            c4 = self.layer3(c3)
            outs.append(c4)
            if max(self.out_layers) >= 4:
                c5 = self.layer4(c4)
                outs.append(c5)
                if max(self.out_layers) >= 5:
                    c6 = self.layer5(c5)
                    outs.append(c6)
        if -1 not in self.out_layers:
            features = [outs[i] for i in self.out_layers]
        else:
            features = [outs[i + 1] for i in self.out_layers]
        return {'features': features, 'strides': self.get_outstrides()}

    def get_outplanes(self):
        return self.out_planes

    def get_outstrides(self):
        return self.out_strides

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def train(self, mode=True):
        """
        Sets the module in training mode.
        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        # self.freeze_layer()
        return self


@MODULE_ZOO_REGISTRY.register('LPCV_2023_Seg_Backbone')
def MetaDetM1_3x3_No_Pool_Single(**kwargs):
    model = MBDet_DepConv3x3_Magic_single(**kwargs, use_maxpool=False)
    return model
