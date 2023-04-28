# MaskCLIP+ with SegFormer
code base : [official MaskCLIP repo](https://github.com/chongzhou96/MaskCLIP), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

This repository contains the implementation and results of an improved version of [MaskCLIP](https://arxiv.org/abs/2112.01071), which incorporates a new classifier that places greater weight on classes predicted by CLIP.

# Zero-shot Segmentation Performance

<table>
    <tr>
        <th>MaskCLIP(RN50)</th>
        <th>mIoU</th>
        <th>config</th>
        <th>json</th>
    </tr>
    <tr>
        <td> base  </td>
        <td>18.46</td>
        <th><a href="https://github.com/dgymjol/MaskCLIP_SegFormer/blob/master/configs/maskclip/maskclip_r50_520x520_pascal_context_59.py">config</a></th>
        <th><a href="https://github.com/dgymjol/MaskCLIP_SegFormer/blob/master/work_dirs/maskclip_r50_520x520_pascal_context_59/eval_single_scale_20230427_211723.json">json</a></th>
    </tr>
    <tr>
        <td> + class weight</td>
        <td> <strong>19.21 </strong></td>
        <th><a href="https://github.com/dgymjol/MaskCLIP_SegFormer/blob/master/configs/maskclip_text/maskclip_text_r50_520x520_pascal_context_59.py">config</a></th>
        <th><a href="https://github.com/dgymjol/MaskCLIP_SegFormer/blob/master/work_dirs/maskclip_text_r50_85prompts/eval_single_scale_20230427_230639.json">json</a></th>
    </tr>
</table>

<table>
    <tr>
        <th>MaskCLIP(ViT16)</th>
        <th>mIoU</th>
        <th>config</th>
        <th>json</th>
    </tr>
    <tr>
        <td> base  </td>
        <td>21.68</td>
        <th><a href="https://github.com/dgymjol/MaskCLIP_SegFormer/blob/master/configs/maskclip/maskclip_vit16_520x520_pascal_context_59.py">config</a></th>
        <th><a href="https://github.com/dgymjol/MaskCLIP_SegFormer/blob/master/work_dirs/maskclip_vit16_520x520_pascal_context_59/eval_single_scale_20230427_214100.json">json</a></th>
    </tr>
    <tr>
        <td> + class weight</td>
        <td> <strong>22.91 </strong></td>
        <th><a href="https://github.com/dgymjol/MaskCLIP_SegFormer/blob/master/configs/maskclip_text/maskclip_text_vit16_520x520_pascal_context_59.py">config</a></th>
        <th><a href="https://github.com/dgymjol/MaskCLIP_SegFormer/blob/master/work_dirs/maskclip_text_vit16_85prompts/eval_single_scale_20230427_222508.json">json</a></th>
    </tr>
</table>

![Data](demo/demo_maskclip_text.png)

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

**Error 0.** ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```shell
sudo apt-get update
sudo apt-get install libgl1
```

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