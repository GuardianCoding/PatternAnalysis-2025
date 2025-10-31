# 🖼️ Greyscale to Colour Image Conversion using MambaIRv2

**Author:** Youssef Hassan (48621573)  
**Course:** COMP3710 – Pattern Analysis (2025)  
**Difficulty:** Hard (Topic Recognition Stream)

---

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Algorithm Description](#algorithm-description)
- [Architecture Visualization](#architecture-visualization)
- [How It Works](#how-it-works)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Training Strategy](#training-strategy)
- [Usage](#usage)
- [Example Results](#example-results)
- [Notes on Preprocessing & Splits](#notes-on-preprocessing--splits)
- [References](#references)
- [Contact](#contact)

---

## Overview

This repository fine-tunes **MambaIRv2**, an advanced **state-space image restoration model**, to perform **grayscale-to-colour conversion**.  
The project adapts pretrained RGB→RGB restoration weights (`mambairv2_ColorDN_15`) for grayscale→RGB mapping using a three-part composite loss: **L1**, **LPIPS**, and **chroma-aware YUV**.  
The approach produces high-quality, perceptually realistic colour reconstructions from single-channel images and is fully reproducible through its provided configuration, datasets, and training scripts.

---

## Problem Statement

Grayscale colourization is a **challenging inverse problem** — one intensity pattern can map to many valid colour combinations. The model must infer context, texture, and semantics to produce convincing colour.  
This project’s objective is to generate accurate and perceptually consistent colours while maintaining structural integrity.  
Challenges addressed:
- Missing chroma data → inferred via learned semantics.  
- Avoiding oversaturation or washed-out colours.  
- Maintaining luminance detail from grayscale input.  

---

## Algorithm Description

The **MambaIRv2** network is an *attentive state-space model* that replaces attention with selective SSM layers to achieve linear-time complexity and long-range spatial awareness.

### Adaptations for Colourization
- **Input/Output mapping:** Replicates grayscale input across 3 channels (Y³) → RGB output.  
- **Loss formulation:** Combines *Charbonnier L1*, *LPIPS perceptual loss*, and *YUV chroma loss*.  
- **λ<sub>UV</sub> Cosine decay:** Dynamically lowers chroma weighting throughout epochs to stabilize training.  
- **Mixed Precision + EMA:** Ensures numerical stability and smooth convergence.  
- **Tiled inference:** Supports large image predictions without exceeding VRAM.

---

## Architecture Visualization

```
Input (Greyscale)
   ↓
Gray → Equalize → Stack (Y³)
   ↓
conv_first
   ↓
Mamba Blocks (6× state-space layers)
   ↓
conv_after_body
   ↓
conv_last
   ↓
Predicted RGB Output
```

**Loss Calculation:**
```
Total Loss = L1 + LPIPS + λ_uv * UV
λ_uv(t) = λ_base * (0.4 + 0.6 * cos_decay(epoch))
```

---

## How It Works

1. **Dataset Handling:** Loads COCO-2017 automatically (train/val split).  
2. **Preprocessing:** Each RGB sample → equalised grayscale input (`gray3`).  
3. **Training Pipeline:**  
   - Uses pretrained MambaIRv2 weights for transfer learning.  
   - Losses are balanced adaptively via cosine scheduling.  
   - Mixed-precision (AMP) and exponential moving averages (EMA) are enabled for stability.  
4. **Validation:** Periodically evaluated using LPIPS, PSNR, and SSIM metrics.  
5. **Logging & Visualization:**  
   - Tracks per-step losses via `StatTracker`.  
   - Produces live `plots.svg` and `logs/train_log.csv`.  
   - Exports comparison panels (`outputs/.../panels`).  

---

## Dependencies

The complete environment is defined in `env.yml`.  
To recreate:
```bash
make setup
conda activate mamba-colour
```

**Key Libraries:**
| Library | Version | Purpose |
|----------|----------|----------|
| PyTorch | ≥2.0.1 | Deep learning core |
| torchvision | ≥0.15 | Image processing |
| basicsr | latest | MambaIRv2 architecture |
| lpips | 0.1 | Perceptual similarity metric |
| pycocotools | latest | COCO dataset tools |
| Pillow | ≥10.0 | Image I/O |
| matplotlib | latest | Training plots |

---

## Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/GuardianCoding/PatternAnalysis-2025.git
cd PatternAnalysis-2025/recognition/topic-recognition
```

### 2️⃣ Setup Environment
```bash
make setup

conda activate mamba-colour
```

### 3️⃣ Download Pretrained Weights
Download [mambairv2_ColorDN_15.pth](https://github.com/csguoh/MambaIR/releases/tag/v1.0) and place it in:
```
checkpoints/mambairv2_ColorDN_15.pth
```

---

## Dataset & Preprocessing

| Step | Description |
|------|--------------|
| **Dataset** | COCO 2017 (`train2017`, `val2017`) |
| **Resize** | Random scale factor between 1.0–1.15× |
| **Crop** | Random 256×256 crop |
| **Flip** | 50% horizontal flip |
| **Augment** | Brightness/contrast jitter |
| **Convert** | RGB → Grayscale → Replicate to 3 channels |

**Train/Validation Split:**  
80% training, 20% validation with deterministic seed (1337).

---

## Training Strategy

Two-stage fine-tuning process for stable adaptation:

| Stage | Description | LR | Loss |
|--------|--------------|------|------|
| Stage 1 | Freeze backbone, train final blocks | 3e-4 | L1 only |
| Stage 2 | Unfreeze all layers | 5e-5 | L1 + LPIPS + UV |

**Training Example:**
```bash
python train.py --config configs/config.yml   --pretrained checkpoints/mambairv2_ColorDN_15.pth   --exp_name mamba_colorizer
```

**Outputs:**
- `outputs/<exp>/plots.svg`  
- `outputs/<exp>/logs/train_log.csv`  
- `outputs/<exp>/panels/`  
- `outputs/<exp>/checkpoints/*.ckpt`  

---

## Usage

**Inference Command:**
```bash
python predict.py --config configs/config.yml   --ckpt outputs/mamba_colorizer_best.ckpt   --test_root ./datasets/test_images   --tile 512 --overlap 32 --amp
```

**Output Directory:**
```
outputs/predict/mamba_colorizer/
├── color/     → Generated colour images
├── panels/    → Comparison grids
├── metrics.csv
└── config_merged.yaml
```

---

## Example Results

### Comparison Panels
| Input (Grayscale) | Prediction (RGB) | Ground Truth |
|:------------------:|:----------------:|:-------------:|
| ![](outputs/predict/panels/example_gray.jpg) | ![](outputs/predict/panels/example_pred.jpg) | ![](outputs/predict/panels/example_gt.jpg) |

**Validation Metrics (Kodak24HQ):**
| Metric | Mean | Median |
|--------:|------:|------:|
| LPIPS ↓ | 0.214 | 0.207 |
| PSNR ↑  | 28.4 dB | 28.1 dB |
| SSIM ↑  | 0.901 | 0.898 |

---

## Notes on Preprocessing & Splits

- Equalized grayscale improves contrast and texture awareness.  
- Cosine decay for λ<sub>UV</sub> balances early chroma learning with late texture refinement.  
- Mixed-precision training improves VRAM efficiency without numerical instability.  
- COCO’s dataset variety enables the model to generalize across lighting and material types.  

---

## References

1. **Guo et al. (2024)** — *MambaIRv2: Attentive State Space Restoration.*  
2. **Zhang et al. (2018)** — *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS).*  
3. **Lin et al. (2014)** — *Microsoft COCO: Common Objects in Context.*  

---

## Acknowledgements

This repository extends the official **MambaIRv2** implementation with a custom fine-tuning and evaluation pipeline for colour restoration.  
Developed by **Youssef Hassan** for the **COMP3710 Pattern Analysis (2025)** project at **The University of Queensland**.

Special thanks to Dr. Shakes Chandra, Dr Gayan Kulatilleke, and my Runpod.io credits.

---