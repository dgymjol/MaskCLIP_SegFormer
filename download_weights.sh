mkdir -p pretrain

python tools/maskclip_utils/convert_clip_weights.py --model RN50 --backbone
python tools/maskclip_utils/convert_clip_weights.py --model ViT16 --backbone

python tools/maskclip_utils/convert_clip_weights.py --model RN50
python tools/maskclip_utils/convert_clip_weights.py --model ViT16

python tools/maskclip_utils/prompt_engineering.py --model RN50 --class-set context
python tools/maskclip_utils/prompt_engineering.py --model ViT16 --class-set context

# for text module
# python tools/maskclip_utils/generate_text_features.py --model ViT32 --class-set context
# python tools/maskclip_utils/generate_text_features_binary.py --model ViT32 --class-set context
python tools/maskclip_utils/generate_text_features.py --model ViT16 --class-set context
