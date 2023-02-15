# optimizer
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
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=4000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)
