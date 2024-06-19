# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
import torch
from mmseg.registry import MODELS
from ..utils import resize
import numpy as np


class ChannelAttention(nn.Module):
    # 定义init函数；
    # channel表示输入进来的通道数；
    # ratio表示缩放的比例，用于第一次全连接
    def __init__(self, in_channels, ratio=8):
        # 初始化
        super(ChannelAttention, self).__init__()
        # 平均池化，输出高、宽为1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化，输出高、宽为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接，或者用SENet中Sequential模块；
        # Linear和Conv2d有什么区别？
        self.seq = nn.Sequential()
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)

        # 相加后再取sigmoid
        self.sigmoid = nn.Sigmoid()

    # 前传部分，out * x 放到最后
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# 空间注意机制不需要传入通道数和ratio，但有卷积核大小3或者7
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # padding = 7整除2 = 3
        padding = 3 if kernel_size == 7 else 1

        # 输入通道为2，即一层最大池化，一层平均池化
        # 输出通道为1；步长为1，即不需要压缩宽高
        self.seq = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False),
            nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False),
            )
        self.sigmoid = nn.Sigmoid()

    # 前传部分，out * x 放到最后
    def forward(self, x):
        # 在通道上进行最大池化和平均池化
        # 对于pytorch，其通道在第一维度，也就是batch_size之后，dim=1
        # 保留通道，所以keepdim=ture
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 堆叠
        x_ = torch.cat([avg_out, max_out], dim=1)
        # 取卷积
        x_ = self.seq(x_)
        x = x * self.sigmoid(x_)
        return x


# 结合空间和通道注意力机制
class cbam_block(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=8, kernel_size=3):
        super(cbam_block, self).__init__()
        self.channelAttention = ChannelAttention(
            in_channels,
            ratio=ratio)
        self.spatialAttention = SpatialAttention(
            kernel_size=kernel_size)

        padding = 3 if kernel_size == 7 else 1

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False),
            nn.SyncBatchNorm(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            nn.SyncBatchNorm(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = x * self.channelAttention(x)
        x = x * self.spatialAttention(x)
        return x


class Tri_Attention(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 ):

        self.attention = nn.ModuleList()

        for _ in np.arange(3):
            block = cbam_block(in_channels, ratio=8, kernel_size=3)
            self.attention.append(block)

    def forward(self, x):
        x0 = self.attention[0](x)

        tmp = x.permute(0, 2, 1, 3) #batch, h, c, w
        x1 = self.attention[1](tmp)
        x1 = x1.permute(0, 2, 1, 3)

        tmp = x.permute(0, 3, 2, 1) #batch, w, h, c
        x2 = self.attention[2](tmp)
        x2 = x2.permute(0, 3, 2, 1)

        return tuple(x0, x1, x2)


@MODELS.register_module()
class FPN_Multi_Attention(BaseModule):
    """Feature Pyramid Network.

    This neck is the implementation of `Feature Pyramid Networks with tri-attention for Object


    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.attentions = nn.ModuleList([nn.ModuleList() for i in np.arange(3)])
        self.fuses = nn.ModuleList([nn.ModuleList() for i in np.arange(3)])
        self.upsamples = nn.ModuleList([nn.ModuleList() for i in np.arange(3)])

        for i in range(self.start_level, self.backbone_end_level):

            in_channels_ = in_channels[i] + self.out_channels if i < self.backbone_end_level - 1 else in_channels[i]

            for j in range(3):
                #(batch, c, h, w): 提升i+1输出尺寸；融合；降维；注意力块
                upsample = nn.Upsample(scale_factor=2, mode='bilinear') if i != self.backbone_end_level - 1 else None
                fuse = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels_, out_channels=self.out_channels,
                                          kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features=self.out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels,
                              kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features=self.out_channels),
                    nn.ReLU(inplace=True)
                )
                atten_block = SpatialAttention(kernel_size=3)
                self.upsamples[j].append(upsample)
                self.fuses[j].append(fuse)
                self.attentions[j].append(atten_block)


    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        out0, out1, out2 = None, None, None
        for i in range(len(self.in_channels)-1, -1, -1):

            #input for tri-perspectives
            input_ = inputs[i]


            #enlarge the size of output of i + 1 layer
            out0 = self.upsamples[0][i](out0) if out0 is not None else None
            out1 = self.upsamples[1][i](out1) if out1 is not None else None
            out2 = self.upsamples[2][i](out2) if out2 is not None else None

            #fusing与降维
            input0_ = torch.cat([input_, out0], dim=1) if out0 is not None else input_
            input1_ = torch.cat([input_, out1], dim=1) if out1 is not None else input_
            input2_ = torch.cat([input_, out2], dim=1) if out2 is not None else input_

            input0 = self.fuses[0][i](input0_)
            input1 = self.fuses[1][i](input1_)
            input2 = self.fuses[2][i](input2_)

            #注意力: b, c, h, w
            out0 = self.attentions[0][i](input0) + input0

            # b, h, c, w
            input1 = input1.permute(0, 2, 1, 3)
            out1 = self.attentions[1][i](input1) + input1
            out1 = out1.permute(0, 2, 1, 3)

            # b, h, w, c
            input2 = input2.permute(0, 2, 3, 1)
            out2 = self.attentions[2][i](input2) + input2
            out2 = out2.permute(0, 3, 1, 2)

        return tuple([out0, out1, out2])




