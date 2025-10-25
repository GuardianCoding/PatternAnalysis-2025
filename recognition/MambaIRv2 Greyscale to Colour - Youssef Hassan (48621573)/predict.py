#!/usr/bin/env python3
"""
predict.py — inference for grayscale→color using a trained MambaIRv2 model.

Reads settings from your existing config.yml (same file used for training).
You can override test/ckpt paths via CLI flags if you want.

Examples:
  # Basic inference (inputs may be gray or RGB)
  python predict.py --config config.yml --ckpt outputs/mamba_color_lab/best_lpips.ckpt

  # Override test root (if not already in config.yml) and provide GT for metrics
  python predict.py --config config.yml \
    --test_root datasets/ColorDN/Kodak24HQ \
    --gt_root   datasets/ColorDN/Kodak24HQ \
    --ckpt      outputs/mamba_color_lab/best_lpips.ckpt \
    --amp
"""

import os
import argparse
from glob import glob
from pathlib import Path
from datetime import datetime
import yaml
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torchvision.utils import save_image, make_grid
from skimage.metrics import structural_similarity as ssim_metric
import pandas as pd

from modules import build_mambairv2_colorizer
from utils.metrics import lab_to_rgb, lpips_loss, psnr as psnr_fn


# ------------------ helpers ------------------

def load_config(path: str) -> dict:
    cfg = yaml.safe_load(open(path, "r"))
    return cfg

def merge_cfg_cli(cfg: dict, args) -> dict:
    # allow CLI to override
    if args.test_root is not None:
        cfg.setdefault("predict", {})
        cfg["predict"]["test_root"] = args.test_root
    if args.gt_root is not None:
        cfg.setdefault("predict", {})
        cfg["predict"]["gt_root"] = args.gt_root
    if args.ckpt is not None:
        cfg["resume"] = args.ckpt  # reuse same key as training "resume"/or pass explicitly below
    if args.embed_dim is not None:
        cfg.setdefault("model", {})
        cfg["model"]["embed_dim"] = args.embed_dim
    if args.depths is not None and len(args.depths) > 0:
        cfg.setdefault("model", {})
        cfg["model"]["depths"] = args.depths
    if args.amp:
        cfg["amp"] = True
    return cfg

def collect_images(root: str):
    exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.PNG","*.JPG","*.JPEG","*.BMP")
    files = []
    for e in exts:
        files += glob(os.path.join(root, "**", e), recursive=True)
    return sorted(files)

def rgb_pil_to_gray3_tensor(rgb_pil: Image.Image) -> torch.Tensor:
    """PIL RGB -> (1,3,H,W) grayscale replicated to 3 channels, in [0,1]."""
    gray3_pil = F.rgb_to_grayscale(rgb_pil, num_output_channels=3)
    x = T.ToTensor()(gray3_pil).unsqueeze(0)  # (1,3,H,W)
    return x

def tensor01_to_uint8_img(t: torch.Tensor) -> np.ndarray:
    arr = (t.squeeze(0).permute(1,2,0).clamp(0,1).cpu().numpy() * 255.0).round().astype(np.uint8)
    return arr

def ssim_on_tensors(a01: torch.Tensor, b01: torch.Tensor) -> float:
    a = tensor01_to_uint8_img(a01)
    b = tensor01_to_uint8_img(b01)
    return float(ssim_metric(a, b, channel_axis=2, data_range=255))

def load_ckpt_into(model: torch.nn.Module, ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] loaded: {ckpt_path}\n       missing={len(missing)}, unexpected={len(unexpected)}, strict=False")


# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yml used for training")
    ap.add_argument("--ckpt",   type=str, default=None, help="Path to trained checkpoint (.ckpt)")
    ap.add_argument("--test_root", type=str, default=None, help="Folder of images to colorize (overrides config)")
    ap.add_argument("--gt_root",   type=str, default=None, help="Optional GT folder for metrics (match by filename)")
    ap.add_argument("--embed_dim", type=int, default=None, help="Override model.embed_dim if needed")
    ap.add_argument("--depths",    type=int, nargs="+", default=None, help="Override model.depths if needed")
    ap.add_argument("--amp", action="store_true", help="Enable mixed-precision inference")
    args = ap.parse_args()

    # Load + merge config
    cfg = load_config(args.config)
    cfg = merge_cfg_cli(cfg, args)

    # Resolve paths
    test_root = cfg.get("predict", {}).get("test_root", None)
    gt_root   = cfg.get("predict", {}).get("gt_root", None)
    ckpt_path = cfg.get("resume", None) or cfg.get("pretrained", None) or args.ckpt

    if test_root is None:
        raise RuntimeError("No test_root provided. Add predict.test_root to config.yml or pass --test_root.")

    if ckpt_path is None:
        raise RuntimeError("No checkpoint provided. Pass --ckpt or set 'resume' in config.yml to your trained .ckpt.")

    # Build timestamped run folder under out_dir/predict/<exp>/YYYYmmdd_HHMMSS
    out_base = Path(cfg.get("out_dir", "outputs"))
    exp_name = cfg.get("exp_defaults", "exp")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_base / "predict" / exp_name / stamp
    color_dir = run_dir / "color"
    panel_dir = run_dir / "panels"
    run_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    # Save merged config for provenance
    with open(run_dir / "config_merged.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    # Gather inputs
    files = collect_images(test_root)
    if len(files) == 0:
        raise RuntimeError(f"No images found under: {test_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model shell and load trained ckpt
    embed_dim = cfg.get("model", {}).get("embed_dim", 174)
    depths    = cfg.get("model", {}).get("depths", [4,4,6,4])
    net = build_mambairv2_colorizer(
        embed_dim=embed_dim,
        depths=tuple(depths),
        pretrained=None,
        device=device
    ).eval()

    load_ckpt_into(net, ckpt_path)

    have_gt = gt_root is not None and os.path.isdir(gt_root)
    lpips_list, psnr_list, ssim_list, names = [], [], [], []

    autocast_ctx = torch.cuda.amp.autocast if (cfg.get("amp", False) and device.type == "cuda") else torch.cpu.amp.autocast
    with torch.inference_mode(), autocast_ctx():
        for i, p in enumerate(files, 1):
            name = os.path.basename(p)
            # Load input image (RGB), then convert to 3-ch grayscale for the model
            rgb_pil = Image.open(p).convert("RGB")
            x_in = rgb_pil_to_gray3_tensor(rgb_pil).to(device)  # (1,3,H,W)
            pred_rgb = net(x_in).clamp(0, 1)                    # (1,3,H,W) in [0,1]

            # Save colorized image
            save_image(pred_rgb, color_dir / name)

            # Build panel
            imgs = [x_in.clamp(0,1), pred_rgb]
            titles = ["Gray", "Pred"]

            # Optional metrics vs GT
            if have_gt:
                gt_path = os.path.join(gt_root, name)
                if os.path.isfile(gt_path):
                    gt_pil = Image.open(gt_path).convert("RGB")
                    gt = T.ToTensor()(gt_pil).unsqueeze(0).to(device)

                    # Size match by min-crop if needed
                    H = min(gt.shape[2], pred_rgb.shape[2])
                    W = min(gt.shape[3], pred_rgb.shape[3])
                    gt = gt[:, :, :H, :W]
                    pr = pred_rgb[:, :, :H, :W]
                    gx = x_in[:, :, :H, :W]

                    lp = float(lpips_loss(pr, gt).item())
                    ps = float(psnr_fn(pr, gt))
                    ss = float(ssim_on_tensors(pr, gt))
                    lpips_list.append(lp); psnr_list.append(ps); ssim_list.append(ss); names.append(name)

                    imgs = [gx, pr, gt]
                    titles = ["Gray", "Pred", "GT"]
                    print(f"[{i:04d}/{len(files)}] {name}  LPIPS={lp:.4f}  PSNR={ps:.2f}  SSIM={ss:.4f}")
                else:
                    print(f"[{i:04d}/{len(files)}] {name}  (no GT match)")
            else:
                print(f"[{i:04d}/{len(files)}] {name}  saved")

            panel = make_grid(torch.cat(imgs, dim=0), nrow=len(imgs))
            save_image(panel, panel_dir / name)

    # Write metrics summary if any
    if len(lpips_list) > 0:
        df = pd.DataFrame({
            "name": names,
            "LPIPS": lpips_list,
            "PSNR": psnr_list,
            "SSIM": ssim_list
        }).sort_values("name")
        csv_path = run_dir / "metrics.csv"
        df.to_csv(csv_path, index=False)

        print("\n=== Metrics (matched files) ===")
        print(f"LPIPS: mean={np.mean(lpips_list):.4f}  median={np.median(lpips_list):.4f}")
        print(f"PSNR : mean={np.mean(psnr_list):.2f} dB  median={np.median(psnr_list):.2f} dB")
        print(f"SSIM : mean={np.mean(ssim_list):.4f}  median={np.median(ssim_list):.4f}")
        print(f"Count: {len(lpips_list)} / {len(files)}")
        print(f"Saved per-image metrics → {csv_path}")

    print(f"\nDone. Outputs saved to: {run_dir}")
    print(f"   - Colorized images: {color_dir}")
    print(f"   - Panels:           {panel_dir}")
    print(f"   - Config copy:      {run_dir/'config_merged.yaml'}")

if __name__ == "__main__":
    main()