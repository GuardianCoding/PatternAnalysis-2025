# MambaIRv2 for Grayscale Image Colorization

**Student ID:** s4839921  
**Project:** COMP3710 Pattern Analysis — Grayscale → Color with MambaIRv2 (Hard)

---

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Algorithm Description](#algorithm-description)
- [Architecture Visualization](#architecture-visualization)
- [How It Works](#how-it-works)
- [Dependencies](#dependencies)
  - [Environment](#environment)
  - [Key Python Packages](#key-python-packages)
- [Installation](#installation)
  - [Clone & Setup](#clone--setup)
  - [Pretrained Weights](#pretrained-weights)
  - [Quick Verify](#quick-verify)
- [Dataset & Preprocessing](#dataset--preprocessing)
  - [Auto‑Download COCO2017](#auto-download-coco2017)
  - [Preprocessing Pipeline](#preprocessing-pipeline)
  - [Data Splits & Reproducibility](#data-splits--reproducibility)
- [Training Strategy](#training-strategy)
  - [Two‑Stage Fine‑Tuning](#two-stage-fine-tuning)
  - [Losses & Schedules](#losses--schedules)
  - [Training Commands](#training-commands)
- [Usage](#usage)
  - [Batch Inference](#batch-inference)
  - [Tiled Inference for Large Images](#tiled-inference-for-large-images)
- [Example Results](#example-results)
  - [Input/Output Panels](#inputoutput-panels)
  - [Training Curves](#training-curves)
- [Notes on Pre‑processing, Splits & Justification](#notes-on-pre-processing-splits--justification)
- [References](#references)
- [Contact](#contact)

---

## Overview

This repository fine‑tunes **MambaIRv2**, an attentive **state‑space** image restoration backbone, to **colorize grayscale images**. The model starts from the official `mambairv2_ColorDN_15` weights (RGB→RGB restoration) and is adapted for grayscale→RGB colorization with a combined loss (Charbonnier L1 + LPIPS + YUV‑chroma). The training/eval pipeline supports **AMP**, **EMA**, **cosine LR**, per‑epoch **subset sampling**, automatic **COCO‑2017** download, and **metrics/panel** export for qualitative/quantitative assessment.

---

## Problem Statement

**Grayscale colorization** is ill‑posed: many valid chroma assignments map to the same luminance. The model must infer **semantics** (e.g., sky, grass, skin), maintain **textures**, avoid **bleeding**, and yield **plausible** colors. We leverage MambaIRv2 to capture long‑range dependencies efficiently while keeping compute linear in image size.

---

## Algorithm Description

- **Backbone:** MambaIRv2 (vision SSM). Linear‑time selective state‑space layers replace quadratic attention while preserving long‑range context.
- **Adaptation:** Accept 3‑ch **gray³** inputs (luminance replicated/perturbed) and predict 3‑ch **RGB** outputs.
- **Losses:**  
  - **Charbonnier L1** for structure,  
  - **LPIPS (VGG)** for perceptual realism,  
  - **YUV chroma loss** (U,V) with epoch‑wise **λ_uv** decay for stable color learning.
- **Training niceties:** mixed precision, EMA, warmup‑cosine LR, epoch‑wise subset sampling from a fixed pool for faster iteration, frequent metric logging & panel export.

---

## Architecture Visualization

```
Gray (L) → gray³ stack + equalise/jitter
           │
           ▼
   conv_first  ───────────────┐
           │                  │
         Mamba blocks (×N)    │  (state-space modeling, linear-time, windowed)
           │                  │
     conv_after_body          │  (+ residual)
           ▼                  │
        conv_last  ◀──────────┘
           ▼
       RGB output
```

**Loss pipeline (per step):**
```
pred_rgb ── Charbonnier L1 ─┐
                             ├─ total = w_L1·L1 + w_LP·LPIPS + λ_uv(e)·UV
pred_rgb, gt_rgb ── LPIPS ──┤
YUV(pred), YUV(gt) ── |U−U*|+|V−V*| ── UV
```

---

## How It Works

1. **Data**: COCO‑2017 images are randomly resized, cropped to `crop_size` (multiple of 16), flipped, and lightly **RGB‑jittered**. A grayscale **equalisation + tiny RGB dither** (training only) breaks channel identity, nudging the model away from just copying gray.
2. **Model**: Build the official MambaIRv2 and load the ColorDN‑15 weights. Train heads + last K blocks (stage‑1), then unfreeze all (stage‑2).
3. **Training loop**: AMP + AdamW + Warmup‑Cosine, per‑epoch subset sampling for faster, diversified steps, optional EMA, periodic **validation** and **panel** snapshots. All logs/plots/checkpoints go under `outputs/<exp>/...`.
4. **Inference**: Single‑folder batch inference with optional **tiled** forward to handle very large images on limited VRAM; saves **colorized images**, **comparison panels**, and (if GT provided) **LPIPS/PSNR/SSIM** CSV.

---

## Dependencies

### Environment
- Linux (CUDA). Tested with Python **3.10** and CUDA **11.8** via Conda.
- Provided: `env.yml` (Conda), `install_mambair.sh`, and a `Makefile` for one‑shot setup.

### Key Python Packages
- **torch / torchvision** (GPU) — core DL and transforms  
- **basicsr** — provides the official `MambaIRv2` arch  
- **lpips** — perceptual loss (VGG)  
- **pycocotools** — COCO API  
- **Pillow, numpy, matplotlib, scikit‑image** — I/O, plotting, SSIM  
- **yaml** — config loading

> Exact versions are pinned in `env.yml`. Use the Conda environment for reproducibility.

---

## Installation

### Clone & Setup
```bash
git clone https://github.com/GuardianCoding/PatternAnalysis-2025.git
cd PatternAnalysis-2025/recognition/topic-recognition

# (Option A) Makefile one‑liner
make setup

# (Option B) Conda manually
conda env create -f env.yml
conda activate mamba-colour
```

### Pretrained Weights
Download **ColorDN‑15** weights (from the official MambaIR release) and place them in `checkpoints/`:
```bash
mkdir -p checkpoints
# e.g., mambairv2_ColorDN_15.pth
```

### Quick Verify
```bash
python -c "import torch, lpips; print('CUDA:', torch.cuda.is_available()); print('✓ setup ok')"
```

---

## Dataset & Preprocessing

### Auto‑Download COCO2017
Training/eval loaders can **auto‑download & extract** COCO‑2017 (train/val + annotations) to `./datasets/coco` if not present. Override roots via `--train_root/--val_root/--ann_root` in CLI or in `config.yml`.

### Preprocessing Pipeline
- **Resize**: random long‑side scaling (≈1.0–1.15×), min‑side >= `crop_size`  
- **Crop**: random `crop_size × crop_size` (default 256, divisible by 16)  
- **Flip**: horizontal with p=0.5  
- **RGB‑jitter**: brightness/contrast/saturation small jitter with configurable prob/strength  
- **Gray input**: RGB→L, gamma equalise; during training add tiny per‑channel noise and replicate to 3‑ch (**gray³**)  
- **Targets**: full‑color RGB tensors in `[0,1]`

### Data Splits & Reproducibility
- **Pool sampling**: build a fixed random **pool** (e.g., 10k samples) from COCO‑train once; each epoch draws a different subset (e.g., 5k) from this pool — strong variety with stable runtime.
- **Validation**: deterministic subset from COCO‑val; metrics logged periodically.
- **Seed**: global seed set (e.g., `1337`). All run configs, merged config, and command are saved under `outputs/...` for full provenance.

---

## Training Strategy

### Two‑Stage Fine‑Tuning
- **Stage‑1** (few epochs): freeze backbone **except** heads + last *K* blocks. Higher LR (e.g., `3e-4`) to quickly adapt color heads.
- **Stage‑2**: unfreeze **all** layers; lower LR (e.g., `5e-5`) with warmup‑cosine to refine globally.

Other knobs (from `config.yml`): `batch_size`, `grad_accum`, `epoch_subset_size`, `train_pool_size`, `plot_redraw_every`, `panel_every`, `val_every`, `ema_decay`, etc.

### Losses & Schedules
- **Charbonnier L1** (robust L1) on RGB.
- **LPIPS** on resized tensors (e.g., side ≤ 192) for stability.
- **UV chroma loss**: L1 on U/V channels in YUV space.
- **λ_uv scheduling**: cosine or dynamic decay over **epochs**; start high to push color learning, then decay to stabilize structure/perceptual terms.
- **LR**: Warmup‑Cosine scheduler; **AMP** enabled; optional **EMA** weight tracking.

### Training Commands
**Minimal (uses `config.yml`):**
```bash
python train.py --config config.yml --pretrained checkpoints/mambairv2_ColorDN_15.pth --exp_name mamba_colorizer
```

**Override dataset roots (if not using auto‑download):**
```bash
python train.py --config config.yml \
  --train_root /data/coco/train2017 \
  --val_root   /data/coco/val2017   \
  --ann_root   /data/coco/annotations \
  --pretrained checkpoints/mambairv2_ColorDN_15.pth \
  --exp_name mamba_colorizer
```

> Outputs:  
> `outputs/<exp>_<timestamp>/` with `plots.svg`, `logs/*.csv`, periodic `panels/step_*.jpg`, and checkpoints (`best_total.ckpt`, `step_*.ckpt`). A convenience symlink/text pointer `<exp>_latest` is created for the most recent run.

---

## Usage

### Batch Inference
```bash
python predict.py --config config.yml \
  --ckpt outputs/mamba_colorizer_best.ckpt \
  --test_root /path/to/grayscale_or_rgb_images \
  --amp
```
- Accepts RGB or grayscale inputs; internally converts to **gray³**.
- Saves **colorized** images, **comparison panels**, and optional metrics (if `--gt_root` provided with filename matches).
- A merged `config_merged.yaml` is written for reproducibility.

### Tiled Inference for Large Images
```bash
python predict.py --config config.yml \
  --ckpt outputs/mamba_colorizer_best.ckpt \
  --test_root ./my_large_images \
  --tile 512 --overlap 32 --amp
```
- Processes large images in overlapping tiles to reduce VRAM.
- Optional: `--max-side` or `--max-pixels` to pre‑downscale inputs safely.

---

## Example Results

### Input/Output Panels
| Input (Gray) | Prediction (RGB) | Ground Truth |
|:------------:|:----------------:|:------------:|
| ![placeholder](./images/examples/gray_1.png) | ![placeholder](./images/examples/pred_1.png) | ![placeholder](./images/examples/gt_1.png) |
| ![placeholder](./images/examples/gray_2.png) | ![placeholder](./images/examples/pred_2.png) | ![placeholder](./images/examples/gt_2.png) |

> Panels are auto‑saved during training/validation and inference under `outputs/.../panels/`.

### Training Curves
![Training Curves Placeholder](./images/plots/training_curves_placeholder.png)

> Live plot `plots.svg` is updated during training and includes **L1**, **LPIPS**, **UV**, **Total**, **LR (scaled)**, and **λ_uv (scaled)** traces.

---

## Notes on Pre‑processing, Splits & Justification

- **Equalised Gray³ Inputs + Dither:** improves learning signal by avoiding a trivial identity mapping from gray to output RGB, nudging the network to explore chroma hypotheses rather than copying luminance.
- **RGB Jitter (low strength):** encourages robustness to exposure and saturation variance without destabilizing chroma learning.
- **Two‑Stage Schedule:** quickly adapts heads and late blocks, then refines entire backbone — a practical compromise between speed and full fine‑tuning stability.
- **Epoch Subset from Fixed Pool:** provides good sample diversity per epoch while keeping step time predictable and logs stable; combined with a deterministic seed and persisted merged configs this supports **reproducibility**.
- **Validation Choice:** COCO‑val subset is used periodically to monitor overfitting; total loss (weighted sum) is used to pick `best_total.ckpt` as it correlates better with visual quality in colorization than LPIPS alone.

---

## References

1. **Guo, C., et al. (2024).** *MambaIRv2: Attentive State Space Restoration.*  
   arXiv:2411.15269 — [https://arxiv.org/abs/2404.13670](https://arxiv.org/abs/2411.15269)
2. **Zhang, R., et al. (2018).** *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.*  
   CVPR 2018 — [https://github.com/richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)
3. **Lin, T.-Y., et al. (2014).** *Microsoft COCO: Common Objects in Context.*  
   ECCV 2014 — [https://cocodataset.org](https://cocodataset.org)

---

## Acknowledgements

This repository extends the official **MambaIRv2** implementation with a custom fine-tuning and evaluation pipeline for colour restoration.  
Developed by **Youssef Hassan** for the **COMP3710 Pattern Analysis (2025)** project at **The University of Queensland**.

Special thanks to Dr. Shakes Chandra, Dr Gayan Kulatilleke, and my Runpod.io credits.

---

## Contact

**Student ID**: s4839921  
**Course**: COMP3710 — Pattern Analysis  
**Institution**: The University of Queensland (UQ)  
**Year**: 2025

For questions or issues, please open a GitHub Issue on this repository.