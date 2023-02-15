####################################################################
###          Annotation-free Segmentation Performace             ###
####################################################################

GPUS=$2

# # clip-vit & seg-cnn(deeplabv2) - in paper, mIoU = 23.9
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
bash tools/dist_train.sh configs/maskclip_plus/anno_free/maskclip_plus_vit16_deeplabv2_r101-d8_480x480_4k_pascal_context_59.py 2 --work-dir work_dirs/anno_free/vit-dlv2
# python tools/test.py configs/maskclip_plus/anno_free/maskclip_plus_vit16_deeplabv2_r101-d8_480x480_4k_pascal_context_59.py work_dirs/maskclip_plus_vit16_deeplabv2_r101-d8_480x480_4k_pascal_context_59/latest.pth --show-dir output/anno_free_vit_deeplabv2

# clip-cnn(r50) & seg-cnn(deeplabv2) - in paper, mIoU = 23.9
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
bash tools/dist_train.sh configs/maskclip_plus/anno_free/maskclip_plus_r50_deeplabv2_r101-d8_480x480_4k_pascal_context_59.py 2 --work-dir work_dirs/anno_free/r50-dl2
# python tools/test.py configs/maskclip_plus/anno_free/maskclip_plus_r50_deeplabv2_r101-d8_480x480_4k_pascal_context_59.py work_dirs/maskclip_plus_r50_deeplabv2_r101-d8_480x480_4k_pascal_context_59/latest.pth --show-dir output/anno_free_r50_deeplabv2



# # # clip-vit & seg-vit(segformerb0)
bash tools/dist_train.sh configs/maskclip_plus/anno_free/maskclip_plus_vit16_segformer_b0_480x480_4k_pascal_context_59.py 2 --cfg-options optimizer.lr=0.00012 data.samples_per_gpu=4 --work-dir work_dirs/anno_free/vit-sfb0-4k-base
lrs=(0.0006 0.0012 0.06 0.00006 0.000012 0.000006)
for lr in ${lrs[@]}
    do 
        bash tools/dist_train.sh configs/maskclip_plus/anno_free/maskclip_plus_vit16_segformer_b0_480x480_4k_pascal_context_59.py 2 --cfg-options optimizer.lr=${lr} data.samples_per_gpu=4 --work-dir work_dirs/anno_free/vit-sfb0-4k-${lr}-4
    done

# clip-cnn(r50) & seg-vit(segformerb0)
bash tools/dist_train.sh configs/maskclip_plus/anno_free/maskclip_plus_r50_segformer_b0_480x480_4k_pascal_context_59.py 2 --cfg-options optimizer.lr=0.00012 data.samples_per_gpu=4 --work-dir work_dirs/anno_free/r50-sfb0-4k-base
lrs=(0.0006 0.0012 0.06 0.00006 0.000012 0.000006)
for lr in ${lrs[@]}
    do 
        bash tools/dist_train.sh configs/maskclip_plus/anno_free/maskclip_plus_r50_segformer_b0_480x480_4k_pascal_context_59.py 2 --cfg-options optimizer.lr=${lr} data.samples_per_gpu=4 --work-dir work_dirs/anno_free/r50-sfb0-4k-${lr}-4
    done



# # # clip-vit & seg-vit(segformerb5)
# bash tools/dist_train.sh configs/maskclip_plus/anno_free/maskclip_plus_vit16_segformer_b5_480x480_8k_pascal_context_59.py 2

# # clip-cnn(r50) & seg-vit(segformerb5)
# bash tools/dist_train.sh configs/maskclip_plus/anno_free/maskclip_plus_r50_segformer_b5_480x480_4k_pascal_context_59.py 2



# ####################################################################
# ###              Zero-shot Segmentation Performace               ###
# ####################################################################

# GPUS=$2
# # # clip-vit & seg-cnn(deeplabv3)
# # default : data.samples_per_gpu=4 | optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
# bash tools/dist_train.sh configs/maskclip_plus/zero_shot/maskclip_plus_vit16_deeplabv3plus_r101-d8_480x480_40k_pascal_context.py 2 --work-dir work_dirs/zero_shot/vit_cnn --cfg-options optimizer.lr=0.002 optimizer.weight_decay=0.00005 data.samples_per_gpu=2

# # clip-cnn(r50) & seg-cnn(deeplabv3) - org
# # default : data.samples_per_gpu=4 | optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
# bash tools/dist_train.sh configs/maskclip_plus/zero_shot/maskclip_plus_r50_deeplabv3plus_r101-d8_480x480_40k_pascal_context.py 2 --work-dir work_dirs/zero_shot/cnn_cnn --cfg-options optimizer.lr=0.002 optimizer.weight_decay=0.00005 data.samples_per_gpu=2

# # # clip-vit & seg-vit(segformerb5) - small lr
# bash tools/dist_train.sh configs/maskclip_plus/zero_shot/maskclip_plus_vit16_segformer_b5_480x480_40k_pascal_context.py --work-dir work_dirs/zero_shot/vit_vit_small --cfg-options optimizer.lr=0.0000012 optimizer.weight_decay=0.001

# # clip-cnn(r50) & seg-vit(segformerb5) - small lr
# bash tools/dist_train.sh configs/maskclip_plus/zero_shot/maskclip_plus_r50_segformer_b5_480x480_40k_pascal_context.py --work-dir work_dirs/zero_shot/cnn_vit_small --cfg-options optimizer.lr=0.0000024 optimizer.weight_decay=0.002 data.samples_per_gpu=4


# # clip-vit & seg-vit(segformerb5) - big lr
# bash tools/dist_train.sh configs/maskclip_plus/zero_shot/maskclip_plus_vit16_segformer_b5_480x480_40k_pascal_context.py --work-dir work_dirs/zero_shot/vit_vit_big --cfg-options optimizer.lr=0.00003 optimizer.weight_decay=0.005

# # # clip-cnn(r50) & seg-vit(segformerb5) - big lr
# bash tools/dist_train.sh configs/maskclip_plus/zero_shot/maskclip_plus_r50_segformer_b5_480x480_40k_pascal_context.py --work-dir work_dirs/zero_shot/cnn_vit_big --cfg-options optimizer.lr=0.00006 optimizer.weight_decay=0.01 data.samples_per_gpu=4

bash tools/dist_train.sh configs/maskclip_plus/zero_shot/maskclip_plus_vit16_segformer_b0_480x480_40k_pascal_context.py
