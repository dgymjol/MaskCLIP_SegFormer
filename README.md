# MaskCLIP+ with SegFormer
code base : [official MaskCLIP repo](https://github.com/chongzhou96/MaskCLIP), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

This repository contains an implementation of [MaskCLIP+](https://arxiv.org/abs/2112.01071) that uses the [Segformer](https://arxiv.org/abs/2105.15203) backbone instead of DeepLabv2

# Annotation-Free Segmentation Performance

<table>
    <tr>
        <th>CLIP backbone</th>
        <th>Segmentor</th>
        <th>mIoU</th>
        <th>Total Params</th>
        <th>config</th>
        <th>log</th>
    </tr>
    <tr>
        <td rowspan=2>CLIP(ResNet50)</td>
        <td>  DeepLabv2-ResNet101 </td>
        <td> <strong>24.82 </strong></td>
        <td> 156M </td>
        <th><a href="https://github.com/dgymjol/MaskCLIP_SegFormer/blob/master/work_dirs/anno_free/r50-dl2/maskclip_plus_r50_deeplabv2_r101-d8_480x480_4k_pascal_context_59.py">config</a></th>
        <th><a href="https://github.com/dgymjol/MaskCLIP_SegFormer/blob/master/work_dirs/anno_free/r50-dl2/20230220_004900.log">log</a></th>
    </tr>
    <tr>
        <td>SegFormer-b5</td>
        <td> 22.87</td>
        <td> 125M</td>
        <th><a href="https://github.com/dgymjol/MaskCLIP_SegFormer/blob/master/work_dirs/anno_free/r50-sfb5/maskclip_plus_r50_segformer_b5_480x480_8k_pascal_context_59.py">config</a></th>
        <th><a href="https://github.com/dgymjol/MaskCLIP_SegFormer/blob/master/work_dirs/anno_free/r50-sfb5/20230219_195921.log">log</a></th>
    </tr>
    <tr>
        <td rowspan=2>CLIP(ViT16)</td>
        <td>  DeepLabv2-ResNet101 </td>
        <td> 31.56 </td>
        <td> 166M </td>
        <th><a href="https://github.com/dgymjol/MaskCLIP_SegFormer/blob/master/work_dirs/anno_free/vit-dlv2/maskclip_plus_vit16_deeplabv2_r101-d8_480x480_4k_pascal_context_59.py">config</a></th>
        <th><a href="https://github.com/dgymjol/MaskCLIP_SegFormer/blob/master/work_dirs/anno_free/vit-dlv2/20230219_231252.log">log</a></th>
    </tr>
    <tr>
        <td>SegFormer-b5</td>
        <td> <strong>33.88</strong></td>
        <td> 169M</td>
        <th><a href="https://github.com/dgymjol/MaskCLIP_SegFormer/blob/master/work_dirs/anno_free/vit-sfb5/maskclip_plus_vit16_segformer_b5_480x480_8k_pascal_context_59.py">config</a></th>
        <th><a href="https://github.com/dgymjol/MaskCLIP_SegFormer/blob/master/work_dirs/anno_free/vit-sfb5/20230219_124812.log">log</a></th>
    </tr>
</table>

![Data](demo.png)

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

**Step 4.** Convert pretrained mit models to MMSegmentation style
```shell
python tools/model_converters/mit2mmseg.py pretrain/mit_b0.pth pretrain/mit_b0_weight.pth
```

# MaskCLIP+

MaskCLIP+ trains another segmentation model(SegFormer) with pseudo labels extracted from MaskCLIP.

**Train.** (please refer to [train.md](docs/en/train.md)

```shell
# if single GPUs, (examples in exp_1.sh)
python tools/train.py ${CONFIG_FILE}

# if multiple GPUs, (examples in exp_2.sh)
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

# Troubleshooting

**Error 1.** `ImportError: MagickWand shared library not found.`
```shell
sudo apt-get update
sudo apt-get install libmagickwand-dev
```

**Error 2.** `ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version GLIBCXX_3.4.29 not found `
```shell
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get install --only-upgrade libstdc++6
```

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