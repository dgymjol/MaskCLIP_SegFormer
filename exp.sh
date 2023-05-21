# # MaskCLIP(RN50) + Text Module
python tools/test.py configs/maskclip_text/maskclip_text_r50_520x520_pascal_context_59.py pretrain/RN50_clip_backbone.pth --eval mIoU --work-dir work_dirs/maskclip-text-vit16/r50  --show-dir output/maskclip-text-vit16/r50

# # MaskCLIP(ViT50) + Text Module
python tools/test.py configs/maskclip_text/maskclip_text_vit16_520x520_pascal_context_59.py pretrain/ViT16_clip_backbone.pth --eval mIoU --work-dir work_dirs/maskclip-text-vit16/vit16 --show-dir output/maskclip-text-vit16/vit16


####################################################################
###          Annotation-free Segmentation Performace             ###
####################################################################

# # clip-vit & seg-cnn(deeplabv2) + text module
# org : gpu_nums = 4(in papers) samples_per_gpu=4, optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
python tools/train.py configs/maskclip_plus/anno_free/maskclip_plus_vit16_deeplabv2_r101-d8_class_weight_480x480_4k_pascal_context_59.py --cfg-options data.samples_per_gpu=8  optimizer.lr=0.005 optimizer.weight_decay=0.00025 --work-dir work_dirs/anno_free/vit-dlv2-text-vit16 
python tools/test.py configs/maskclip_plus/anno_free/maskclip_plus_vit16_deeplabv2_r101-d8_class_weight_480x480_4k_pascal_context_59.py work_dirs/anno_free/vit-dlv2-text-vit16/latest.pth --show-dir output/anno_free/vit-dlv2-text-vit16

# clip-cnn(r50) & seg-cnn(deeplabv2) + text module
# org : gpu_nums = 4(in papers) samples_per_gpu=4 optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
python tools/train.py configs/maskclip_plus/anno_free/maskclip_plus_r50_deeplabv2_r101-d8_class_weight_480x480_4k_pascal_context_59.py --cfg-options data.samples_per_gpu=8 optimizer.lr=0.005 optimizer.weight_decay=0.00025 --work-dir work_dirs/anno_free/r50-dl2-text-vit16
python tools/test.py configs/maskclip_plus/anno_free/maskclip_plus_r50_deeplabv2_r101-d8_class_weight_480x480_4k_pascal_context_59.py work_dirs/anno_free/r50-dl2-text-vit16/latest.pth --show-dir output/anno_free/r50-dl2-text-vit16
