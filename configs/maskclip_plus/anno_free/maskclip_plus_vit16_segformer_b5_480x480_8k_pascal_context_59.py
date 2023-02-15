_base_ = [
    '../../_base_/models/maskclip_plus_vit16_segformer.py', '../../_base_/datasets/pascal_context_59.py', 
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_8k_adamw.py'
]

suppress_labels = list(range(0, 59))
model = dict(
    pretrained='pretrain/mit_b5.pth',
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 6, 40, 3]),
    decode_head=dict(
        num_classes=59,
        text_categories=59,
        text_embeddings_path='pretrain/context_ViT16_clip_text.pth',
        clip_unlabeled_cats=suppress_labels,
        decode_module_cfg=dict(in_channels=[64, 128, 320, 512])
    )
)

find_unused_parameters=True
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (520, 520)
crop_size = (480, 480)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True, suppress_labels=suppress_labels),
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
    samples_per_gpu=2,
    train=dict(
        pipeline=train_pipeline
    )
)