dataset_type = 'SD_saliency_900'
data_root = '../data/SD-saliency-900/'

img_scale = (256, 256)
crop_size = (224, 224)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale),
    dict(type='Normal'),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ignore_index=1,
        data_prefix=dict(
            img_path='images/sp0.0',
            seg_map_path='annotations/validation'),
        test_mode=True,
        pipeline=test_pipeline))