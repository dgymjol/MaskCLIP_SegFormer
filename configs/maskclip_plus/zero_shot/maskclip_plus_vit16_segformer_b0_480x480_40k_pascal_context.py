_base_ = [
    '../../_base_/models/maskclip_plus_vit16_segformer.py', '../../_base_/datasets/pascal_context.py', 
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_40k_adamw.py'
]
# cow,motorbike,sofa,cat,boat,fence,bird,tvmonitor,keyboard,aeroplane
suppress_labels=[20, 34, 49, 15, 8, 25, 7, 55, 32, 1]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='/root/clip/MaskCLIP/pretrain/SegFormer_B0.pth',
    decode_head=dict(
        text_categories=59, 
        text_embeddings_path='pretrain/context_ViT16_clip_text.pth',
        clip_unlabeled_cats=suppress_labels,
        unlabeled_cats=suppress_labels,
        start_clip_guided=(1, 3999),
        start_self_train=(4000, -1),
        cls_bg=True,
        decode_module_cfg=dict(
            type='SegformerHead',
            in_channels=[32, 64, 160, 256],
            in_index=[0, 1, 2, 3],
            channels=512,
            dropout_ratio=0.1,
            num_classes=59,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
            )
        ),
    )
)

find_unused_parameters=True
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (520, 520)
crop_size = (480, 480)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', suppress_labels=suppress_labels),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
data = dict(
    samples_per_gpu=8,
    train=dict(
        pipeline=train_pipeline
    )
)
# optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
optimizer = dict(
                type='AdamW',
                    lr=6e-05,
                    betas=(0.9, 0.999),
                    weight_decay=0.01,
                    paramwise_cfg=dict(
                        custom_keys=dict(
                            pos_block=dict(decay_mult=0.0),
                            norm=dict(decay_mult=0.0),
                            head=dict(lr_mult=10.0)
                        )
                    )
                )