####################################################################
###              Zero-shot Segmentation Performace               ###
####################################################################

GPUS=$2
# # clip-vit & seg-cnn(deeplab)
# default : data.samples_per_gpu=4 | optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
bash tools/dist_train.sh configs/maskclip_plus/zero_shot/maskclip_plus_vit16_deeplabv3plus_r101-d8_480x480_40k_pascal_context.py 2 --work-dir work_dirs/vit_cnn --cfg-options optimizer.lr=0.002 optimizer.weight_decay=0.00005 data.samples_per_gpu=2

# clip-cnn(r50) & seg-cnn(deeplab) - org
# default : data.samples_per_gpu=4 | optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
bash tools/dist_train.sh configs/maskclip_plus/zero_shot/maskclip_plus_r50_deeplabv3plus_r101-d8_480x480_40k_pascal_context.py 2 --work-dir work_dirs/cnn_cnn --cfg-options optimizer.lr=0.002 optimizer.weight_decay=0.00005 data.samples_per_gpu=2

# # clip-vit & seg-vit(segformerb5) - small lr
python tools/train.py configs/maskclip_plus/zero_shot/maskclip_plus_vit16_segformer_b5_480x480_40k_pascal_context.py --work-dir work_dirs/vit_vit_small --cfg-options optimizer.lr=0.0000012 optimizer.weight_decay=0.001

# clip-cnn(r50) & seg-vit(segformerb5) - small lr
python tools/train.py configs/maskclip_plus/zero_shot/maskclip_plus_r50_segformer_b5_480x480_40k_pascal_context.py --work-dir work_dirs/cnn_vit_small --cfg-options optimizer.lr=0.0000024 optimizer.weight_decay=0.002 data.samples_per_gpu=4


# clip-vit & seg-vit(segformerb5) - big lr
python tools/train.py configs/maskclip_plus/zero_shot/maskclip_plus_vit16_segformer_b5_480x480_40k_pascal_context.py --work-dir work_dirs/vit_vit_big --cfg-options optimizer.lr=0.00003 optimizer.weight_decay=0.005

# # clip-cnn(r50) & seg-vit(segformerb5) - big lr
python tools/train.py configs/maskclip_plus/zero_shot/maskclip_plus_r50_segformer_b5_480x480_40k_pascal_context.py --work-dir work_dirs/cnn_vit_big --cfg-options optimizer.lr=0.00006 optimizer.weight_decay=0.01 data.samples_per_gpu=4


