# Morphology-Guided Image-Text Alignment for Herbarium Specimen Classification

This repository contains the training code for our herbarium specimen classification framework. The method uses morphology-aware text descriptions during training to improve visual classification, while keeping image-only inference at test time.

The current implementation provides a unified training script for multiple visual backbones and multiple alignment losses. It is designed for controlled experiments, ablation studies, and reproducible comparison across model families.

## Overview

Herbarium specimen classification is difficult because of limited training images, class imbalance, high visual similarity among taxa, and large variation across collections. To address this, we use a multimodal training framework that combines specimen images with morphology-related text descriptions. During training, the model learns both the classification objective and the image-text alignment objective. During inference, the model only needs the image.

The code supports:

- multiple image backbones
- optional image-text alignment
- instance-level and prototype-level alignment losses
- learnable temperature
- class-balanced loss
- balanced sampling
- test-time augmentation
- optional gradient clipping
- deterministic training settings for reproducibility

## Main Features

- Unified training script for different backbones
- Shared data interface based on JSONL files
- Optional decoupled classification and alignment projections
- Optional shared image projection for v3-style behaviour
- Multiple alignment modes:
  - `none`
  - `inst`
  - `inst_dcl`
  - `proto`
  - `proto_dcl`
- Optional support for:
  - ResNet-50
  - ConvNeXt-Base
  - ConvNeXt-Tiny
  - Swin-T
  - Swin-S
  - Swin-B
  - CLIP
  - BioCLIP

## Repository Structure

A recommended repository structure is shown below.

```text
.
├── README.md
├── requirements.txt
├── train_sacm_min_plus_v4_final_tiny.py
├── configs/
│   ├── resnet.yaml
│   ├── convnext.yaml
│   └── swin.yaml
├── scripts/
│   ├── train_resnet.sh
│   ├── train_convnext.sh
│   └── train_swin.sh
├── data/
│   └── example_jsonl/
├── figures/
├── outputs/
└── checkpoints/
```

You may change this layout later. The current README only assumes that the main training script is available.

## Environment

We recommend Python 3.10 or 3.11 and a recent CUDA-enabled PyTorch installation.

Example setup:

```bash
conda create -n herbarium_align python=3.11
conda activate herbarium_align
pip install -r requirements.txt
```

A typical `requirements.txt` should include at least:

```text
torch
torchvision
transformers
numpy
pillow
open_clip_torch
```

If you use BioCLIP or CLIP backbones, please make sure `open_clip_torch` is installed correctly.

## Data Format

The training and test data use a JSONL format. Each line should contain one sample:

```json
{"image": "path/to/image.jpg", "label_id": 0, "text": "morphological description here"}
```

Fields:

- `image`: path to the image file
- `label_id`: integer class index
- `text`: optional morphology-related description used during training

A separate `label2id.json` file is also required. Example:

```json
{
  "Species_A": 0,
  "Species_B": 1,
  "Species_C": 2
}
```

## Training

### Basic example

```bash
python train_sacm_min_plus_v4_final_tiny.py \
  --arch resnet \
  --train_jsonl data/train.jsonl \
  --test_jsonl data/test.jsonl \
  --label2id data/label2id.json \
  --epochs 50 \
  --batch_size 32 \
  --optim adamw \
  --base_lr 3e-4 \
  --weight_decay 5e-4 \
  --head cosine \
  --augment \
  --amp \
  --tta \
  --save
```

### Example with image-text alignment

```bash
python train_sacm_min_plus_v4_final_tiny.py \
  --arch convnext \
  --train_jsonl data/train.jsonl \
  --test_jsonl data/test.jsonl \
  --label2id data/label2id.json \
  --epochs 50 \
  --batch_size 32 \
  --optim adamw \
  --base_lr 3e-4 \
  --weight_decay 5e-4 \
  --head cosine \
  --align_type inst \
  --w_align 0.2 \
  --warmup_align_epochs 8 \
  --temp_init 0.07 \
  --augment \
  --balanced_sampler \
  --cb_loss \
  --amp \
  --tta \
  --save
```

### Example with prototype alignment

```bash
python train_sacm_min_plus_v4_final_tiny.py \
  --arch swin \
  --train_jsonl data/train.jsonl \
  --test_jsonl data/test.jsonl \
  --label2id data/label2id.json \
  --epochs 50 \
  --batch_size 32 \
  --optim adamw \
  --base_lr 3e-4 \
  --weight_decay 5e-4 \
  --head cosine \
  --align_type proto_dcl \
  --w_align 0.25 \
  --warmup_align_epochs 12 \
  --proto_recompute_every 1 \
  --temp_init 0.07 \
  --augment \
  --amp \
  --tta \
  --save
```

## Important Arguments

### Backbone and representation

- `--arch`: visual backbone
- `--cls_proj_dim`: projection dimension for classification branch
- `--align_dim`: projection dimension for alignment branch
- `--share_img_proj`: use a shared image projection for both classification and alignment

### Optimisation

- `--optim`: `adamw` or `sgd`
- `--base_lr`: base learning rate
- `--weight_decay`: weight decay
- `--head_lr_mult`: learning rate multiplier for projection and classifier layers

### Alignment

- `--align_type`: `none`, `inst`, `inst_dcl`, `proto`, or `proto_dcl`
- `--w_align`: alignment loss weight
- `--warmup_align_epochs`: number of warm-up epochs before alignment starts
- `--align_ramp_epochs`: optional ramp schedule for alignment weight
- `--align_backbone_grad_scale`: scale factor for alignment gradients flowing into the image branch
- `--temp_init`: initial temperature value
- `--min_text_tokens`: minimum valid text length for alignment

### Sampling and classification

- `--balanced_sampler`: enable class-balanced sampling
- `--cb_loss`: enable class-balanced classification loss
- `--cb_beta`: beta used in class-balanced weighting
- `--label_smoothing`: label smoothing value
- `--head`: `linear` or `cosine`

### Augmentation and evaluation

- `--augment`: enable training augmentation
- `--amp`: enable automatic mixed precision
- `--tta`: enable test-time augmentation by horizontal flip

### Gradient clipping

- `--grad_clip`: enable gradient clipping
- `--no_grad_clip`: disable gradient clipping
- `--grad_clip_val`: clipping threshold

## Output Files

During training, the script can save:

- model checkpoints
- per-epoch CSV logs
- per-epoch JSONL logs
- summary JSONL files for best results

These logs include training loss, classification loss, alignment loss, accuracy, macro-F1, and other run information.

## Reproducibility

The script includes deterministic training options, fixed random seed control, and structured logging. For fair comparison, we recommend reporting:

- the exact command used
- the random seed
- the backbone
- the alignment type
- the final or best checkpoint selection rule

## Notes

- This repository currently focuses on training and controlled experiments.
- The test set may also be used as the validation set in some experimental settings. Please adjust this part if you prepare a cleaner public release.
- Some paths in the original internal code may need to be changed to relative paths before public release.
- If the dataset cannot be fully released, please provide either a sample subset or instructions for data preparation.

## Citation

If you use this code, please cite our paper.

```bibtex
@article{yourpaper2026,
  title   = {Your Paper Title},
  author  = {Author One and Author Two and Author Three},
  journal = {Under review},
  year    = {2026}
}
```

## Licence

Please add a licence file before public release. MIT or Apache-2.0 are common choices for research code, but the final choice should follow your data and model redistribution conditions.

## Contact

For questions about the code or the project, please open an issue in this repository.

