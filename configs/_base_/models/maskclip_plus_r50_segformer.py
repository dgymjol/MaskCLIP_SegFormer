# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='pretrain/mit_b0.pth',
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='MaskClipPlusSegformerHead',
        vit=False,
        in_channels=2048,
        channels=1024,
        num_classes=0,
        dropout_ratio=0,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        decode_module_cfg=dict(
            type='SegformerHead',
            in_channels=[32, 64, 160, 256],
            in_index=[0, 1, 2, 3],
            channels=1024,
            dropout_ratio=0.1,
            num_classes=59,
            norm_cfg=norm_cfg,
            align_corners=False,
        ),
        text_categories=59,
        text_channels=1024,
        text_embeddings_path='pretrain/context_RN50_clip_text.pth',
        cls_bg=False,
        norm_feat=False,
        clip_unlabeled_cats=list(range(0, 20)),
        clip_cfg=dict(
            type='ResNetClip',
            depth=50,
            norm_cfg=norm_cfg,
            contract_dilation=True
        ),
        clip_weights_path='pretrain/RN50_clip_weights.pth',
        reset_counter=True,
        start_clip_guided=(1, -1),
        start_self_train=(-1, -1)
    ),
    feed_img_to_decode_head=True,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
