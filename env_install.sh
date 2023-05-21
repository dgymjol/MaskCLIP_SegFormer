# conda create --name maskclip python=3.8 -y 
# conda activate maskclip
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
conda install git wget tensorboard -y
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# mmcv install sol 1
pip install -U openmim
pip install mim
mim install mmcv-full==1.5.0

# mmcv install sol 2
# pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install setuptools==59.5.0
pip install pandas gpustat matplotlib numpy
pip install packaging prettytable scikit-image Wand
pip install -v -e .


