# model settings: Steel Surface Defect Detection using Tri-Attention and Hierarchical Feature Fusion
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0.4669, 0.4669, 0.4669],
    std=[0.2437, 0.2437, 0.2437],
    bgr_to_rgb=True,
    pad_val=0,
    size=(224, 224),
    seg_pad_val=1)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV2c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    neck=dict(
        type='FPN_Multi_Attention',
        in_channels=[64, 256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    decode_head=dict(
        type='FPNHead_SSSD',
        in_channels=[256, 256, 256],
        in_index=[0, 1, 2],
        feature_strides=[2, 2, 2],
        channels=128,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
