# coding=utf-8
import torch
import torch.nn as nn
from mmseg.registry import MODELS


def _iou(pred, target, size_average=True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0, b):
        # compute the IoU of the foreground
        Iand1 = torch.sum(target[i, :, :, :]*pred[i, :, :, :])
        Ior1 = torch.sum(target[i, :, :, :])+torch.sum(pred[i, :, :, :])-Iand1
        IoU1 = Iand1/Ior1

        # IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b


@MODELS.register_module()
class IOU(nn.Module):
    def __init__(self, size_average=True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)
