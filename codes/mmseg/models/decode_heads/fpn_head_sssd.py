# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from ..losses import accuracy, IOU, SSIM
from ..utils import Upsample, resize
from .decode_head import BaseDecodeHead
from typing import List, Tuple
from torch import Tensor
from mmseg.utils import ConfigType, SampleList
import torch.nn.functional as F

# ------- 1. define loss function --------
bce_loss = nn.BCELoss(size_average=True)
ssim_loss = SSIM(window_size=11, size_average=True)
iou_loss = IOU(size_average=True)

#1. all = bce_ + ssim_ + iou_ (OK)
#2. bce_ (OK)
#3. ssim_ (OK)
#4. iou_ (OK)
#5. bce_ + ssim_ (OK)
#6. bce_ + iou_ (OK)
#7. ssim_ + iou_ (OK)
def train_loss(pred, target):
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)
    loss = bce_out + ssim_out + iou_out

    return loss

def muti_loss_fusion(preds, target):
    loss_tar = train_loss(preds[0], target)
    loss1 = train_loss(preds[1], target)
    loss2 = train_loss(preds[2], target)
    loss3 = train_loss(preds[3], target)
    return loss_tar, loss1 + loss2 + loss3

@MODELS.register_module()
class FPNHead_SSSD(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

        #target prediction
        self.conv_segs = nn.ModuleList()
        conv_seg = nn.Sequential(
            ConvModule(
                self.channels * 3,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=self.align_corners),
            ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            nn.Conv2d(
                self.channels,
                self.channels,
                3,
                padding=1,
                ),
            nn.Conv2d(in_channels=self.channels, out_channels=self.out_channels, kernel_size=1, padding=0)
        )
        self.conv_segs.append(conv_seg)

        # sub-target prediction
        for i in range(len(self.feature_strides)):
            conv_seg = nn.Sequential(
                nn.Conv2d(self.channels, self.out_channels, kernel_size=1),
                Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=self.align_corners)
            )
            self.conv_segs.append(conv_seg)
        self.conv_seg = self.conv_segs[0]

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        seg_logits = [resize(
            input=seg_logits[i],
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners) for i in range(len(seg_logits))]

        seg_logits_ = [F.sigmoid(seg_logits[i]) for i in range(len(seg_logits))]

        loss_tar, loss_oths = muti_loss_fusion(seg_logits_, seg_label.float())

        loss['loss_tar'] = loss_tar
        loss['loss_oths'] = loss_oths
        loss['acc_seg'] = accuracy(
            seg_logits[0], seg_label.squeeze(1), ignore_index=self.ignore_index)

        return loss

    def forward(self, inputs):

        x = self._transform_inputs(inputs)
        outputs = [self.scale_heads[i](x[i]) for i in range(len(self.feature_strides))]
        output = outputs[0]

        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = torch.cat([output, outputs[i]], dim=1)

        if self.training:
            outputs.insert(0, output)
            output = tuple(outputs)

        output = self.cls_seg(output)
        return output

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            if self.training:
                feat = tuple([self.dropout(feat[i]) for i in range(len(feat))])
            else:
                feat = self.dropout(feat)
        if self.training:
            output = tuple([self.conv_segs[i](feat[i]) for i in range(len(feat))])
        else:
            output = self.conv_segs[0](feat)
        return output

