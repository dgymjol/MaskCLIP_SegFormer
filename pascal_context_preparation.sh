# Prepare data
mkdir data
cd data

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
tar -xvf VOCtrainval_03-May-2010.tar
cd VOCdevkit/VOC2010
wget https://codalabuser.blob.core.windows.net/public/trainval_merged.json

cd ../../
install Detail API
git clone https://github.com/zhanghang1989/detail-api
pip install Cython
cd detail-api/PythonAPI
python setup.py build_ext install

cd ../../../
python tools/convert_datasets/pascal_context.py data/VOCdevkit data/VOCdevkit/VOC2010/trainval_merged.json
