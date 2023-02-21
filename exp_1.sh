GPUS=$1
SAMPLES_PER_GPU=$4 # Defalut

# #####################################################################
# ###               MaskCLIP Segmentation Performace                ###
# #####################################################################

# MaskCLIP(RN50)
python tools/test.py configs/maskclip/maskclip_r50_520x520_pascal_context_59.py pretrain/RN50_clip_backbone.pth --eval mIoU 
python tools/test.py configs/maskclip/maskclip_r50_520x520_pascal_context_59.py pretrain/RN50_clip_backbone.pth --eval mIoU --show-dir output/maskclip/r50

# MaskCLIP(ViT50)
python tools/test.py configs/maskclip/maskclip_vit16_520x520_pascal_context_59.py pretrain/ViT16_clip_backbone.pth --eval mIoU
python tools/test.py configs/maskclip/maskclip_vit16_520x520_pascal_context_59.py pretrain/ViT16_clip_backbone.pth --eval mIoU --show-dir output/maskclip/vit16

# MaskCLIP(RN50) + Text Module
python tools/test.py configs/maskclip_text/maskclip_text_r50_520x520_pascal_context_59.py pretrain/RN50_clip_backbone.pth --eval mIoU --work-dir work_dirs/maskclip_text_r50_85prompts
python tools/test.py configs/maskclip_text/maskclip_text_r50_520x520_pascal_context_59.py pretrain/RN50_clip_backbone.pth --eval mIoU --work-dir work_dirs/maskclip_text_r50_85prompts  --show-dir output/maskclip_text/r50

# MaskCLIP(ViT50) + Text Module
python tools/test.py configs/maskclip_text/maskclip_text_vit16_520x520_pascal_context_59.py pretrain/ViT16_clip_backbone.pth --eval mIoU --work-dir work_dirs/maskclip_text_vit16_85prompts
python tools/test.py configs/maskclip_text/maskclip_text_vit16_520x520_pascal_context_59.py pretrain/ViT16_clip_backbone.pth --eval mIoU --work-dir work_dirs/maskclip_text_vit16_85prompts --show-dir output/maskclip_text/vit16


####################################################################
###          Annotation-free Segmentation Performace             ###
####################################################################

# # clip-vit & seg-cnn(deeplabv2) - in paper, mIoU = 23.9
# org : gpu_nums = 4(in papers) samples_per_gpu=4, optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
python tools/train.py configs/maskclip_plus/anno_free/maskclip_plus_vit16_deeplabv2_r101-d8_480x480_4k_pascal_context_59.py --cfg-options data.samples_per_gpu=8  optimizer.lr=0.005 optimizer.weight_decay=0.00025 --work-dir work_dirs/anno_free/vit-dlv2 
python tools/test.py configs/maskclip_plus/anno_free/maskclip_plus_vit16_deeplabv2_r101-d8_480x480_4k_pascal_context_59.py work_dirs/anno_free/vit-dlv2/latest.pth --show-dir output/anno_free/vit-dlv2

# clip-cnn(r50) & seg-cnn(deeplabv2) - in paper, mIoU = 23.9
# org : gpu_nums = 4(in papers) samples_per_gpu=4 optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
python tools/train.py configs/maskclip_plus/anno_free/maskclip_plus_r50_deeplabv2_r101-d8_480x480_4k_pascal_context_59.py --cfg-options data.samples_per_gpu=8 optimizer.lr=0.005 optimizer.weight_decay=0.00025 --work-dir work_dirs/anno_free/r50-dl2
python tools/test.py configs/maskclip_plus/anno_free/maskclip_plus_r50_deeplabv2_r101-d8_480x480_4k_pascal_context_59.py work_dirs/anno_free/r50-dl2/latest.pth --show-dir output/anno_free/r50-dl2


# # # clip-vit & seg-vit(segformerb5)
# org(pred..) : gpu_nums = 8, samples_per_gpu = 2, optimizer = dict(type='Adamw', lr=0.000024, momentum=0.9, weight_decay=0.004)
python tools/train.py configs/maskclip_plus/anno_free/maskclip_plus_vit16_segformer_b5_480x480_8k_pascal_context_59.py --cfg-options data.samples_per_gpu=8 optimizer.lr=0.000024 optimizer.weight_decay=0.004 --work-dir work_dirs/anno_free/vit-sfb5
python tools/test.py configs/maskclip_plus/anno_free/maskclip_plus_vit16_segformer_b5_480x480_8k_pascal_context_59.py work_dirs/anno_free/vit-sfb5/latest.pth --show-dir output/anno_free/vit-sfb5


# clip-cnn(r50) & seg-vit(segformerb5)
# org(pred..) : gpu_nums = 8, samples_per_gpu = 2, optimizer = dict(type='Adamw', lr=0.000024, momentum=0.9, weight_decay=0.004)
python tools/train.py configs/maskclip_plus/anno_free/maskclip_plus_r50_segformer_b5_480x480_8k_pascal_context_59.py --cfg-options data.samples_per_gpu=6 optimizer.lr=0.00006 optimizer.weight_decay=0.004 --work-dir work_dirs/anno_free/r50-sfb5
python tools/test.py configs/maskclip_plus/anno_free/maskclip_plus_r50_segformer_b5_480x480_8k_pascal_context_59.py work_dirs/anno_free/r50-sfb5/latest.pth --show-dir output/anno_free/r50-sfb5



######### supplementary ########
lrs=(0.006 0.0012 0.0006 0.00012 0.00006 0.000012 0.000024)

# # # clip-vit & seg-vit(segformerb0)
for lr in ${lrs[@]}
    do 
        python tools/train.py configs/maskclip_plus/anno_free/maskclip_plus_vit16_segformer_b0_480x480_8k_pascal_context_59.py --cfg-options optimizer.lr=${lr} data.samples_per_gpu=16 --work-dir work_dirs/anno_free/vit-sfb0-${lr}
    done

# # #clip-cnn(r50) & seg-vit(segformerb0)
for lr in ${lrs[@]}
    do 
        python tools/train.py configs/maskclip_plus/anno_free/maskclip_plus_r50_segformer_b0_480x480_8k_pascal_context_59.py --cfg-options optimizer.lr=${lr} data.samples_per_gpu=16 --work-dir work_dirs/anno_free/r50-sfb0-${lr}
    done

# TODO: Zero-Shot Segmentation