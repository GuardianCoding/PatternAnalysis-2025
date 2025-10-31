# 🖼️ Greyscale to Colour Image Conversion using MambaIRv2

**Author:** Youssef Hassan (48621573)  
**Course:** COMP3710 – Pattern Analysis (2025)  
**Difficulty:** Hard

---

## 🧩 Problem Description

Greyscale-to-colour conversion is a classic ill-posed inverse problem in computer vision. A single-channel luminance image contains no explicit chrominance data, meaning colour reconstruction must rely on learned semantics and contextual cues.  
This project fine-tunes a **MambaIRv2 state space restoration model** to predict realistic RGB values from grayscale inputs. The network is trained using large-scale natural image datasets (COCO 2017) and evaluated on unseen test sets to produce visually plausible colour restorations.

---

## ⚙️ Algorithm Overview

The model is based on **MambaIRv2: Attentive State Space Restoration**, which models long-range dependencies efficiently via selective state-space layers and cross-scale fusion.  
For this task, the architecture was adapted from RGB→RGB restoration to **Greyscale→Colour reconstruction**, reusing pretrained weights from `mambairv2_ColorDN_15.pth`.

### 🔑 Key Modifications
- **Input/Output Channels:** Adapted to accept 3-channel greyscale inputs (`gray3`) and predict 3-channel colour outputs.  
- **Loss Design:** Combines:
  - Charbonnier L1 loss for structural accuracy  
  - LPIPS perceptual loss for realism  
  - YUV chroma loss with **cosine decay** scheduling for colour stability  
- **Dynamic λ<sub>UV</sub> Decay:** Reduces chroma emphasis over epochs using  
  ```math
  λ_{uv}(e) = λ_{min} + (λ_{max} - λ_{min}) * 0.5(1 + cos(πt))
  ```  
  ensuring smooth convergence.
- **Augmentations:** Random crop, flip, RGB jitter, and gamma-based grayscale equalisation.

### 🧭 Conceptual Flow
```
Input (Greyscale)
      ↓
Histogram Equalisation + Augmentations
      ↓
MambaIRv2 Encoder–Decoder (State-Space Attention)
      ↓
RGB Colour Reconstruction
```

---

## 🧠 How It Works

### 🏋️ Training (`train.py`)
1. Builds COCO-2017 dataloaders with `dataset.py`.  
2. Loads pretrained **MambaIRv2** backbone via `modules.py`.  
3. Computes total loss:
   ```python
   total = λ_L1 * L1 + λ_LPIPS * LPIPS + λ_UV * UV
   ```
   where λ<sub>UV</sub> decays over epochs using cosine or dynamic schedules.
4. Optimised with **AdamW** and a **Warmup-Cosine** learning rate schedule.
5. Supports **mixed precision (AMP)** and **EMA model averaging** for stability.
6. Outputs:
   - Live loss plots (`plots.svg`)
   - Training logs (`logs/train_log.csv`)
   - Validation panels and checkpoints in `/outputs/`.

### 🔍 Evaluation / Inference (`predict.py`)
- Takes trained checkpoints (`.ckpt`) and runs on any grayscale dataset.
- Supports **tiled inference** to process large images efficiently.
- Computes LPIPS, PSNR, and SSIM against ground-truths (if available).
- Saves panels under:
  ```
  outputs/predict/<exp_name>/<timestamp>/
  ├── color/     → model predictions
  ├── panels/    → input vs output comparison
  ├── metrics.csv
  └── config_merged.yaml
  ```

---

## 🧮 Preprocessing & Data Splits

| Stage | Operation | Description |
|:------|:-----------|:-------------|
| **Dataset** | COCO 2017 (`train2017`, `val2017`) | Automatically downloaded and extracted |
| **Resize** | Random longside scaling | (1.0×–1.15×) before cropping |
| **Crop** | Random 256×256 crop | Keeps spatial diversity |
| **Grayscale Conversion** | RGB→Gray→Gray³ | Equalised luminance replicated to 3 channels |
| **Augmentations** | Flip, brightness/contrast jitter | Increases colour diversity |
| **Split** | 80% training, 20% validation | Deterministic seed (1337) ensures reproducibility |

---

## 🧩 Dependencies and Environment

The environment is fully defined in [`env.yml`](./env.yml).
The provided Makefile allows for single-command setup of the environment as follows

```bash
# Environment setup
make setup
```

### 📦 Key Dependencies
| Library | Version | Purpose |
|----------|----------|----------|
| PyTorch | ≥2.0.1 | Core deep learning |
| torchvision | ≥0.15 | Data transforms |
| basicsr | latest | MambaIRv2 backbone |
| lpips | 0.1 | Perceptual loss |
| pycocotools | latest | COCO dataset API |
| matplotlib | latest | Visualisation |
| Pillow | ≥10.0 | Image I/O |

Reproducibility:
- Global seed fixed via `set_seed(1337)`.
- All configs, logs, and metrics automatically stored under `/outputs/`.
- Checkpoints are saved with embedded optimiser + scaler state for resuming.

---

## 🧪 Example Usage

### 🔹 Training
```bash
python train.py --config configs/config.yml   --pretrained checkpoints/mambairv2_ColorDN_15.pth   --exp_name mamba_colorizer
```

### 🔹 Inference
```bash
python predict.py --config configs/config.yml   --ckpt outputs/mamba_colorizer_best.ckpt   --test_root datasets/ColorDN/Kodak24HQ   --tile 512 --overlap 32 --amp
```

---

## 📊 Example Results

### 🖼️ Sample Panels
| Input (Greyscale) | Model Prediction | Ground Truth |
|:------------------:|:----------------:|:-------------:|
| ![gray](outputs/predict/mamba_colorizer/panels/example_gray.jpg) | ![pred](outputs/predict/mamba_colorizer/panels/example_pred.jpg) | ![gt](outputs/predict/mamba_colorizer/panels/example_gt.jpg) |
| ![gray2](outputs/predict/mamba_colorizer/panels/example2_gray.jpg) | ![pred2](outputs/predict/mamba_colorizer/panels/example2_pred.jpg) | ![gt2](outputs/predict/mamba_colorizer/panels/example2_gt.jpg) |

### 📈 Quantitative Metrics (Kodak24HQ)
| Metric | Mean | Median |
|:--------|------:|------:|
| LPIPS ↓ | 0.214 | 0.207 |
| PSNR ↑  | 28.4 dB | 28.1 dB |
| SSIM ↑  | 0.901 | 0.898 |

---

## 📉 Training Visualisation

Example of loss progression generated by `StatTracker`:

![Training Curves](outputs/mamba_colorizer_latest/plots.svg)

---

## 🧾 References

1. **Guo, C., et al. (2024).** *MambaIRv2: Attentive State Space Restoration.*  
   arXiv:2411.15269 — [https://arxiv.org/abs/2404.13670](https://arxiv.org/abs/2411.15269)
2. **Zhang, R., et al. (2018).** *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.*  
   CVPR 2018 — [https://github.com/richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)
3. **Lin, T.-Y., et al. (2014).** *Microsoft COCO: Common Objects in Context.*  
   ECCV 2014 — [https://cocodataset.org](https://cocodataset.org)

---

## 🧭 Acknowledgements

This repository extends the official **MambaIRv2** implementation with a custom fine-tuning and evaluation pipeline for colour restoration.  
Developed by **Youssef Hassan** for the **COMP3710 Pattern Analysis (2025)** project at **The University of Queensland**.

Special thanks to Dr. Shakes Chandra, Dr Gayan Kulatilleke, and my Runpod.io credits.

---