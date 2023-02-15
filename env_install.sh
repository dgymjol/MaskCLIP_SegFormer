conda create --name maskclip python=3.8 -y 
conda activate maskclip
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 -c pytorch -y
conda install git wget tensorboard -y
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install -U openmim
pip install mim
mim install mmcv-full==1.5.0
pip install setuptools==59.5.0
pip install pandas gpustat matplotlib numpy
pip install packaging prettytable scikit-image Wand
pip install -v -e .