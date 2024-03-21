# NeRF-HuGS: Improved Neural Radiance Fields in Non-static Scenes Using Heuristics-Guided Segmentation

[Jiahao Chen](https://cnhaox.github.io), [Yipeng Qin](https://yipengqin.github.io), [Lingjie Liu](https://lingjie0206.github.io), [Jiangbo lu](https://sites.google.com/site/jiangbolu), [Guanbin Li](http://guanbinli.com)

[[`Paper`]()] [[`Project`](https://cnhaox.github.io/NeRF-HuGS/)] [[`Data`](https://drive.google.com/drive/folders/19xmJGgL4VlqviXIDdiy-tLzthu6SJrzn?usp=sharing)] [[`BibTeX`](#Citation)]


## Introduction

This repository contains the code release for **CVPR 2024** paper "NeRF-HuGS: Improved Neural Radiance Fields in Non-static Scenes Using Heuristics-Guided Segmentation".

The codebase mainly consists of three parts:

1. `Mip-NeRF 360`: A fork of the official [MultiNeRF](https://github.com/google-research/multinerf). It has been simplified by removing Ref-NeRF and RawNeRF.
2. `nerfacto`: A re-implementation of Nerfacto and vanilla NeRF referring to the official [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) w/o the pose optimization. 
3. `HuGS`: An implementation of Heuristics-Guided Segmentation based on [COLMAP](https://colmap.github.io/) and [Segment Anything](https://segment-anything.com/).

## Setup

### Environment

In this project, we need **two conda environments**:
1. one with [JAX](https://jax.readthedocs.io/) for `Mip-NeRF 360`;
2. one with [PyTorch](https://pytorch.org/) for `Nerfacto`, `HuGS` and `metrics.py`.

Here's an example of creating environments using GPU with CUDA 11.8.
```bash
# 1. Clone the repo.
git clone https://github.com/cnhaox/NeRF-HuGS --recursive
cd NeRF-HuGS

# 2. Make a conda environment for jax.
conda create -n nerfhugs_jax python=3.9
conda activate nerfhugs_jax

# 3. Install jax, jaxlib and other requirements. 
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements_jax.txt

# 4. Make another conda environment for pytorch.
conda create -n nerfhugs_torch python=3.9
conda activate nerfhugs_torch

# 5. Install pytorch, tiny-cuda-nn, segment anything and other requirements.
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install -r requirements_torch.txt
```

### Dataset

We mainly use three public datasets in the paper:

* **Phototourism Dataset** (proposed by [NeRF-W](https://nerf-w.github.io/)). 
You can download the scenes from [here](https://www.cs.ubc.ca/~kmyi/imw2020/data.html) and official train/test split files from [here](https://nerf-w.github.io/). 
The directory structure should be as follows:
```bash
phototourism
|-- brandenburg_gate
|   |-- dense
|   |   |-- ...
|   |   |-- static_masks # (optional) position of static maps
|   |-- brandenburg.tsv  # data split file
|-- sacre_coeur
|-- taj_mahal
|-- trevi_fountain
```

* **Kubric Dataset** (proposed by [D2NeRF](https://d2nerf.github.io/)). 
You can download the scenes from [here](https://drive.google.com/drive/folders/1B97cgpv3ivYlUXg6yqIAPRnOa4l1Kcx3). 
The directory structure should be as follows:
```bash
kubric
|-- kubric_bag
|   |-- ...
|   |-- static_masks # (optional) position of static maps
|-- kubric_car
|-- kubric_cars
|-- kubric_chairs
|-- kubric_pillow
```

* **Distractor Dataset** (proposed by [RobustNeRF](https://robustnerf.github.io/)). 
You can download it from [here](https://storage.googleapis.com/jax3d-public/projects/robustnerf/robustnerf.tar.gz) and unofficial train/test split files from [here](https://drive.google.com/drive/folders/19xmJGgL4VlqviXIDdiy-tLzthu6SJrzn?usp=sharing). 
The directory structure should be as follows:
```bash
distractor
|-- and-bot
|   |-- 0
|   |   |-- ...
|   |   |-- static_masks    # (optional) position of static maps
|   |   |-- data_split.json # data split file
|-- crab
|-- t_balloon_statue
|-- yoda
```

> **Remark**: `static_masks` is the folder including static maps proposed in our paper.
> You can download our HuGS's results as static maps from [here](https://drive.google.com/drive/folders/19xmJGgL4VlqviXIDdiy-tLzthu6SJrzn?usp=sharing). If `static_masks` does not exist, the dataloader will assume that all pixels are static.

## Running NeRF

Example scripts for training and evaluating NeRF can be found in `scripts/`. You'll need to change the following variables:
*  `JAX_ENV` / `TORCH_ENV`: Environment name of JAX / PyTorch.
* `CUDA_VISIBLE_DEVICES`: GPU IDs for training and evaluating. Mip-NeRF 360 (net_width=1024) may need four GPUs (24 GB VRAM) for training, and Nerfacto may need one.
* `DATA_DIR`: Path of the dataset.
* `SCENES`: Scene names included in `DATA_DIR` for training and evaluating.
* `CONFIG_NAME`: Name of the config file in `<MipNeRF360_or_nerfacto>/configs` without the last suffix. 

[NeRF-W](https://nerf-w.github.io/), [HA-NeRF](https://rover-xingyu.github.io/Ha-NeRF/) (w/o appearance hallucination module) and [RobustNeRF](https://robustnerf.github.io/) have also been re-implemented on all of the baseline models.
Their config files can be found in `<MipNeRF360_or_nerfacto>/configs`.

## Running HuGS

We offer an example script `scripts/hugs_kubric.sh` to train partially trained Nerfacto, generate static maps and visualizations of HuGS for Kubric dataset. 
You may need to change additional variables:
* `COLMAP_DIR`: Path of the COLMAP folder, which directly includes cameras.bin, images.bin and points3D.bin.
* `SAM_TYPE`: SAM model type. Default: "vit_h"
* `SAM_PATH`: Path of the SAM checkpoint. You can download it from [here](https://github.com/facebookresearch/segment-anything/#model-checkpoints).

Since the original Kubric dataset doesn't include COLMAP files, you may need to use COLMAP to process the scenes first.

## Reproducing results

We offer [checkpoints](https://drive.google.com/drive/folders/19xmJGgL4VlqviXIDdiy-tLzthu6SJrzn?usp=sharing) and related [rendering results](https://drive.google.com/drive/folders/19xmJGgL4VlqviXIDdiy-tLzthu6SJrzn?usp=sharing) of Mip-NeRF 360 (base, w/ RobustNeRF, w/ HuGS) and Nerfacto (base, w/ HuGS) to verify the metrics in the paper. 
You can use `scripts/eval_*.sh` to reproduce the rendering results and use `scripts/metrics_*.sh` to calculate PSNR, SSIMs and LPIPS of the images. 
The quantitative results should be consistent with our paper.

> **Remark**: The results of Phototourism Dataset may be slightly different from our paper (~0.01 in PSNR, ~0.001 in SSIM and LPIPS), because the resized ground truths used in our paper are not quantized from float $[0.,1.]$ to uint $[0, 255]$ to be stored.

## Citation

If you find this project useful, please consider citing:

```bibtex
@article{chen2024nerfhugs,
  author    = {Chen, Jiahao and Qin, Yipeng and Liu, Lingjie and Lu, Jiangbo and Li, Guanbin},
  title     = {NeRF-HuGS: Improved Neural Radiance Fields in Non-static Scenes Using Heuristics-Guided Segmentation},
  journal   = {CVPR},
  year      = {2024},
}
```