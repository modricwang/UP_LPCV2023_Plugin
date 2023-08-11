# Import from third library
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .....models.backbones.ac_kernels.ac import D3L_ACBlock
from up.utils.model.normalize import build_conv_norm, build_norm_layer
from up.utils.model.initializer import initialize_from_cfg
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from ...models.backbones.dbb import DiverseBranchBlock

__all__ = ['Deconv_BiFPN']


def build_conv_norm_dbb(in_channels,
                        out_channels,
                        kernel_size,
                        stride=1,
                        padding=0,
                        dilation=1,
                        groups=1,
                        internal_channels_1x1_3x3=-1,
                        normalize={'type': 'solo_bn'},
                        activation=False,
                        relu_first=False,
                        block=None):
    if block == 'DBB':
        conv = DiverseBranchBlock(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, normalize=normalize, internal_channels_1x1_3x3=internal_channels_1x1_3x3)
        return conv
    # for compability
    else:
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=False)
    if (normalize is None) and (not activation):
        return conv

    seq = nn.Sequential()
    if relu_first and activation:
        seq.add_module('relu', nn.ReLU(inplace=True))
    seq.add_module('conv', conv)
    if normalize is not None:
        norm_name, norm = build_norm_layer(out_channels, normalize)
        seq.add_module(norm_name, norm)
    if activation:
        if not relu_first:
            seq.add_module('relu', nn.ReLU(inplace=True))
    return seq


@MODULE_ZOO_REGISTRY.register('LPCV_2023_Seg_Neck')
class Deconv_BiFPN(nn.Module):
    """
    Feature Pyramid Network

    .. note::

        If num_level is larger than backbone's output feature layers, additional layers will be stacked

    """

    def __init__(self,
                 inplanes,
                 outplanes,

                 start_level,
                 num_level,
                 out_strides,
                 downsample,
                 upsample,
                 normalize={'type': 'solo_bn'},
                 align_channels=False,
                 tocaffe_friendly=False,
                 initializer=None,
                 align_corners=True,
                 use_p5=False,
                 num_repeat=1,
                 skip=False,
                 split_deconv=False,
                 use_AC=False,
                 use_dbb=False,
                 dbb_ratio=2.,
                 freeze_square_conv=False,
                 padding=[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
                 output_padding=[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                 deconv_kernel=[4, 4, 4, 4, 4]):
        """
        Arguments:
            - inplanes (:obj:`list` of :obj:`int`): input channel
            - outplanes (:obj:`list` of :obj:`int`): output channel, all layers are the same
            - start_level (:obj:`int`): start layer of backbone to apply FPN, it's only used for naming convs.
            - num_level (:obj:`int`): number of FPN layers
            - out_strides (:obj:`list` of :obj:`int`): stride of FPN output layers
            - downsample (:obj:`str`): method to downsample, for FPN, it's ``pool``, for RetienaNet, it's ``conv``
            - upsample (:obj:`str`): method to upsample, ``nearest`` or ``bilinear``
            - normalize (:obj:`dict`): config of Normalization Layer
            - initializer (:obj:`dict`): config for model parameter initialization

        `FPN example <http://gitlab.bj.sensetime.com/project-spring/pytorch-object-detection/blob/
        master/configs/baselines/faster-rcnn-R50-FPN-1x.yaml#L75-82>`_
        """

        super(Deconv_BiFPN, self).__init__()
        # print('inplanes', inplanes, flush=True)
        assert downsample in ['pool', 'conv'], downsample
        assert isinstance(inplanes, list)
        self.inplanes = inplanes
        if align_channels:
            self.outplanes = outplanes
        else:
            self.outplanes = inplanes
        self.align_channels = align_channels
        self.outstrides = out_strides
        self.start_level = start_level
        self.num_level = num_level
        self.downsample = downsample
        self.upsample = upsample
        self.tocaffe_friendly = tocaffe_friendly
        if upsample == 'nearest':
            align_corners = None
        self.align_corners = align_corners
        self.use_p5 = use_p5
        self.skip = skip
        self.split_deconv = split_deconv
        self.freeze_square_conv = freeze_square_conv
        self.use_dbb = use_dbb
        assert num_level == len(out_strides)
        self.deconvs = []
        self.num_repeat = num_repeat

        if use_dbb:
            block_type = 'DBB'
        else:
            block_type = None

        while len(self.inplanes) < num_level:
            lvl_idx = len(self.inplanes)
            self.inplanes.append(self.inplanes[-1])
            self.add_module(self.get_input_downsample_name(lvl_idx),
                nn.Sequential(
                    build_conv_norm_dbb(self.inplanes[lvl_idx - 1], self.inplanes[lvl_idx], kernel_size=3, stride=2,
                        padding=1,
                        internal_channels_1x1_3x3=inplanes[lvl_idx] * dbb_ratio,
                        normalize=normalize, activation=False, block=block_type),
                    nn.ReLU())
            )
        if align_channels:
            for lvl_idx in range(max(num_level, len(inplanes))):
                if isinstance(self.outplanes, list):
                    outplanes = self.outplanes[lvl_idx]
                else:
                    outplanes = self.outplanes
                self.add_module(self.get_align_name(lvl_idx),
                    nn.Sequential(
                        build_conv_norm_dbb(inplanes[lvl_idx], outplanes, kernel_size=3, stride=1, padding=1,
                            internal_channels_1x1_3x3=inplanes[lvl_idx] * dbb_ratio,
                            normalize=normalize, block=block_type),
                        nn.ReLU())
                )
                self.inplanes[lvl_idx] = outplanes

        for R in range(self.num_repeat):
            for lvl_idx in range(max(num_level, len(inplanes)) - 1):
                # print('inplanes', inplanes, lvl_idx, inplanes[lvl_idx], flush=True)
                # channel = inplanes[lvl_idx]
                self.add_module(
                    self.get_downsample_name(lvl_idx, R),
                    build_conv_norm_dbb(inplanes[lvl_idx], inplanes[lvl_idx + 1], kernel_size=3, stride=2, padding=1,
                        internal_channels_1x1_3x3=inplanes[lvl_idx + 1] * dbb_ratio,
                        normalize=normalize, block=block_type))
                self.add_module(self.get_downsample_relu_name(lvl_idx, R), nn.ReLU())
                if upsample == 'deconv':
                    self.add_module(
                        self.get_deconv_name(lvl_idx, R),
                        nn.ConvTranspose2d(inplanes[lvl_idx + 1],
                            inplanes[lvl_idx],
                            kernel_size=deconv_kernel[lvl_idx],
                            stride=2,
                            padding=padding[lvl_idx],
                            output_padding=output_padding[lvl_idx]))
                elif upsample == 'deconv_and_conv':

                    self.add_module(
                        self.get_deconv_name(lvl_idx, R),
                        nn.ConvTranspose2d(inplanes[lvl_idx + 1],
                            inplanes[lvl_idx],
                            kernel_size=deconv_kernel[lvl_idx],
                            stride=2,
                            padding=padding[lvl_idx],
                            output_padding=output_padding[lvl_idx]))
                    self.add_module(
                        self.get_upsample_name(lvl_idx, R),
                        build_conv_norm_dbb(inplanes[lvl_idx], inplanes[lvl_idx], kernel_size=3, stride=1,
                            padding=1,
                            internal_channels_1x1_3x3=inplanes[lvl_idx] * dbb_ratio,
                            normalize=normalize, block=block_type)
                    )
                elif upsample == 'deconv_relu_and_conv':

                    self.add_module(
                        self.get_deconv_name(lvl_idx, R),
                        nn.ConvTranspose2d(inplanes[lvl_idx + 1],
                            inplanes[lvl_idx],
                            kernel_size=deconv_kernel[lvl_idx],
                            stride=2,
                            padding=padding[lvl_idx],
                            output_padding=output_padding[lvl_idx]))
                    self.add_module(
                        self.get_upsample_name(lvl_idx, R),
                        build_conv_norm_dbb(inplanes[lvl_idx], inplanes[lvl_idx], kernel_size=3, stride=1,
                            padding=1,
                            internal_channels_1x1_3x3=inplanes[lvl_idx] * dbb_ratio,
                            normalize=normalize, block=block_type)
                    )
                    self.add_module(self.get_upsample_relu2_name(lvl_idx, R), nn.ReLU())
                elif upsample == "bilinear_conv" or upsample == "bilinear_conv_after_add":
                    self.add_module(
                        self.get_upsample_name(lvl_idx, R),
                        build_conv_norm_dbb(inplanes[lvl_idx+1], inplanes[lvl_idx], kernel_size=3, stride=1,
                            padding=1,
                            internal_channels_1x1_3x3=inplanes[lvl_idx] * dbb_ratio,
                            normalize=normalize, block=block_type)
                    )
                else:
                    self.add_module(
                        self.get_upsample_name(lvl_idx, R),
                        build_conv_norm_dbb(inplanes[lvl_idx + 1], inplanes[lvl_idx], kernel_size=3, stride=1,
                            padding=1,
                            internal_channels_1x1_3x3=inplanes[lvl_idx] * dbb_ratio,
                            normalize=normalize, block=block_type)
                    )
                self.add_module(self.get_upsample_relu_name(lvl_idx, R), nn.ReLU())
        initialize_from_cfg(self, initializer)

    def get_align_name(self, idx):
        return 'c{}_align'.format(idx + self.start_level)

    def get_align(self, idx):
        return getattr(self, self.get_align_name(idx))

    def get_input_downsample_name(self, idx):
        return 'intput_c{}_align'.format(idx + self.start_level)

    def get_input_downsample(self, idx):
        return getattr(self, self.get_input_downsample_name(idx))

    def get_lateral_name(self, idx, R):
        return 'c{}_r{}_lateral'.format(idx + self.start_level, R)

    def get_lateral(self, idx, R):
        return getattr(self, self.get_lateral_name(idx, R))

    def get_deconv_name(self, idx, R):
        return 'deconv{}_r{}_lateral'.format(idx + self.start_level, R)

    def get_deconv(self, idx, R):
        return getattr(self, self.get_deconv_name(idx, R))

    # def get_upsample(self, idx, R):
    #     if self.upsample == 'deconv':
    #         return getattr(self, self.get_deconv(idx, R))
    #     else:
    #         return

    def get_downsample_name(self, idx, R):
        return 'p{}_r{}_{}'.format(idx + self.start_level, R, self.downsample)

    def get_downsample(self, idx, R):
        return getattr(self, self.get_downsample_name(idx, R))

    def get_downsample_relu_name(self, idx, R):
        return 'p{}_r{}_down_relu'.format(idx + self.start_level, R)

    def get_downsample_relu(self, idx, R):
        return getattr(self, self.get_downsample_relu_name(idx, R))

    def get_upsample_relu_name(self, idx, R):
        return 'p{}_r{}_up_relu'.format(idx + self.start_level, R)

    def get_upsample_relu(self, idx, R):
        return getattr(self, self.get_upsample_relu_name(idx, R))
    def get_upsample_relu2_name(self, idx, R):
        return 'p{}_r{}_up_relu2'.format(idx + self.start_level, R)

    def get_upsample_relu2(self, idx, R):
        return getattr(self, self.get_upsample_relu_name(idx, R))
    def get_upsample_name(self, idx, R):
        return 'p{}_r{}_upsample{}'.format(idx + self.start_level, R, self.downsample)

    def get_upsample(self, idx, R):
        return getattr(self, self.get_upsample_name(idx, R))

    def get_pconv_name(self, idx, R):
        return 'p{}_r{}_conv'.format(idx + self.start_level, R)

    def get_pconv(self, idx, R):
        return getattr(self, self.get_pconv_name(idx, R))

    def forward(self, input):
        """
        .. note::

            - For faster-rcnn, get P2-P5 from C2-C5, then P6 = pool(P5)
            - For RetinaNet, get P3-P5 from C3-C5, then P6 = Conv(C5), P7 = Conv(P6)

        Arguments:
            - input (:obj:`dict`): output of ``Backbone``

        Returns:
            - out (:obj:`dict`):

        Input example::

            {
                'features': [],
                'strides': []
            }

        Output example::

            {
                'features': [], # list of tenosr
                'strides': []   # list of int
            }
        """
        features = input['features']
        # print('----------before----------',flush=True)
        # for item in features:
        #     print(item.shape,flush=True)
        # print('----------end----------', flush=True)
        while len(features) < self.num_level:
            lvl_idx = len(features)
            features.append(self.get_input_downsample(lvl_idx)(features[-1]))
        # print('----------before----------',flush=True)
        # for item in features:
        #     print(item.shape,flush=True)
        # print('----------end----------', flush=True)

        if self.align_channels:
            for lvl_idx in range(len(features)):
                features[lvl_idx] = self.get_align(lvl_idx)(features[lvl_idx])

        for R in range(self.num_repeat):
            laterals = []
            new_features = []
            laterals.append(features[-1])
            for lvl_idx in range(len(self.inplanes) - 1)[::-1]:
                if self.upsample == 'deconv':
                    try:
                        laterals.append(
                            self.get_upsample_relu(lvl_idx, R)(
                                features[lvl_idx] + self.get_deconv(lvl_idx, R)(laterals[-1])))
                    except Exception as e:
                        print(e)
                        print(lvl_idx, R, features[lvl_idx].shape, laterals[-1].shape, self.get_deconv(lvl_idx, R),
                            flush=True)

                        raise e
                elif self.upsample == 'deconv_and_conv':
                    laterals.append(
                        self.get_upsample_relu(lvl_idx, R)(
                            self.get_upsample(lvl_idx, R)(
                                features[lvl_idx] + self.get_deconv(lvl_idx, R)(laterals[-1]))))
                elif self.upsample == 'deconv_relu_and_conv':
                    laterals.append(
                        self.get_upsample_relu(lvl_idx, R)(
                            self.get_upsample(lvl_idx, R)(
                                self.get_upsample_relu2(lvl_idx,R)(
                                features[lvl_idx] + self.get_deconv(lvl_idx, R)(laterals[-1])))))
                elif self.upsample == "bilinear_conv":
                    laterals.append(self.get_upsample_relu(lvl_idx, R)(features[lvl_idx] +
                                                                       self.get_upsample(lvl_idx, R)(
                                                                           F.interpolate(laterals[-1],
                                                                               scale_factor=2,
                                                                               mode="bilinear",
                                                                               align_corners=self.align_corners))))
                elif self.upsample == "bilinear_conv_after_add":
                    laterals.append(self.get_upsample_relu(lvl_idx, R)(
                        self.get_upsample(lvl_idx, R)(
                            features[lvl_idx] +
                            F.interpolate(laterals[-1],
                                scale_factor=2,
                                mode="bilinear",
                                align_corners=self.align_corners))))
                else:
                    laterals.append(self.get_upsample_relu(lvl_idx, R)(features[lvl_idx] +
                                                                       F.interpolate(self.get_upsample(lvl_idx, R)(
                                                                           laterals[-1]),
                                                                           scale_factor=2,
                                                                           mode=self.upsample,
                                                                           align_corners=self.align_corners)))
            laterals = laterals[::-1]
            new_features.append(laterals[0])
            for lvl_idx in range(1, len(self.inplanes) - 1):
                new_features.append(self.get_downsample_relu(lvl_idx, R)(
                    features[lvl_idx] +
                    laterals[lvl_idx] +
                    self.get_downsample(lvl_idx - 1, R)(new_features[-1])))
            last_idx = len(self.inplanes) - 2
            new_features.append(self.get_downsample_relu(last_idx, R)(
                features[-1] + self.get_downsample(last_idx, R)(new_features[-1])))

            # for item in laterals:
            #     if item is not None:
            #         print(item.shape)
            # for item in new_features:
            #     if item is not None:
            #         print(item.shape)
            features = new_features
        return {'features': features, 'strides': self.get_outstrides()}

    def get_outplanes(self):
        """
        Return:
            - outplanes (:obj:`list` of :obj:`int`)
        """
        return self.outplanes

    def get_outstrides(self):
        return torch.tensor(self.outstrides, dtype=torch.int)
