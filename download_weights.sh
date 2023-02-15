mkdir -p pretrain

python tools/maskclip_utils/convert_clip_weights.py --model RN50 --backbone
python tools/maskclip_utils/convert_clip_weights.py --model ViT16 --backbone
python tools/maskclip_utils/convert_clip_weights.py --model RN50
python tools/maskclip_utils/convert_clip_weights.py --model ViT16
python tools/maskclip_utils/prompt_engineering.py --model RN50 --class-set context
python tools/maskclip_utils/prompt_engineering.py --model ViT16 --class-set context