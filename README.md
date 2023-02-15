# MaskCLIP+ with SegFormer

MaskCLIP paper: [Extract Free Dense Labels from CLIP](https://arxiv.org/abs/2112.01071).

Official MaskCLIP code : [MaskCLIP github](https://github.com/chongzhou96/MaskCLIP)

# Setup
**Step 0.**  Make a conda environment
```shell
bash env_install.sh
```

**Step 1.**  Dataset Preparation (ref : [dataset_prepare.md](docs/en/dataset_prepare.md#prepare-datasets))

```shell
bash pascal_context_preparation.sh
```

**Step 2.**  Download and convert the CLIP models & Prepare the text embeddings

```shell
bash download_weights.sh
```

**Step 3.**  Download the SegFormer weights pretrained on ImageNet-1 at [here](https://github.com/NVlabs/SegFormer#trainings) and locate them in `pretrain` folder



# MaskCLIP+

MaskCLIP+ trains another segmentation model(SegFormer) with pseudo labels extracted from MaskCLIP.

**Train.** (please refer to [train.md](docs/en/train.md))

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
```

**Inference.** 

Get quantitative results (mIoU):
```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU
```
Get qualitative results:
```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show-dir ${OUTPUT_DIR}
```

In exp.sh, there are many examples
# Citation
the code base is  MaskCLIP
```
@InProceedings{zhou2022maskclip,
    author = {Zhou, Chong and Loy, Chen Change and Dai, Bo},
    title = {Extract Free Dense Labels from CLIP},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2022}
}
```