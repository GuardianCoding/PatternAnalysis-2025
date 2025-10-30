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

import warnings # Ignore warnings from lpip module
warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*deprecated.*")
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument."
)
warnings.filterwarnings(
    "ignore",
    message="Applied workaround for CuDNN issue, install nvrtc.so"
)

import os
import argparse
from glob import glob
from pathlib import Path
from datetime import datetime
import yaml
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torchvision.utils import save_image, make_grid
from skimage.metrics import structural_similarity as ssim_metric
import pandas as pd
from torch.cuda.amp import autocast
from contextlib import nullcontext

from modules import build_mambairv2_colorizer
from utils import lpips_loss, psnr as psnr_fn


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

def maybe_downscale_pil(img: Image.Image, max_side=0, max_pixels=0) -> Image.Image:
    W, H = img.size
    if max_pixels and H*W > max_pixels:
        s = (max_pixels / (H*W))**0.5
        W, H = max(1, int(W*s)), max(1, int(H*s))
        img = img.resize((W, H), Image.BICUBIC)
    if max_side and max(H, W) > max_side:
        s = max_side / max(H, W)
        W, H = max(1, int(W*s)), max(1, int(H*s))
        img = img.resize((W, H), Image.BICUBIC)
    return img

import math

def pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int,int]]:
    _,_,H,W = x.shape
    Hn = math.ceil(H / multiple) * multiple
    Wn = math.ceil(W / multiple) * multiple
    if (Hn, Wn) == (H, W): 
        return x, (0,0)
    xpad = torch.nn.functional.pad(x, (0, Wn-W, 0, Hn-H), mode="reflect")
    return xpad, (Hn-H, Wn-W)

@torch.no_grad()
def forward_tiled(net, x01, tile=512, overlap=32, pad_mult=8):
    # Optionally pad to model/window multiple to avoid boundary artifacts
    x_pad, (ph, pw) = pad_to_multiple(x01, pad_mult)
    _, C, H, W = x_pad.shape
    out = torch.zeros_like(x_pad)
    norm = torch.zeros((1,1,H,W), device=x_pad.device, dtype=x_pad.dtype)

    step = tile - overlap
    for y in range(0, H, step):
        for x in range(0, W, step):
            y0 = y
            x0 = x
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)
            # grow box to include overlap but clip to image
            y0i = max(0, y1 - tile)
            x0i = max(0, x1 - tile)
            patch = x_pad[:, :, y0i:y1, x0i:x1]
            pred  = net(patch).clamp(0,1)
            out[:, :, y0i:y1, x0i:x1] += pred
            norm[:, :, y0i:y1, x0i:x1] += 1.0

    out = out / norm.clamp_min(1.0)
    # unpad back to original size
    if ph or pw:
        out = out[:, :, :H - ph, :W - pw]
    return out

def save_panel_with_titles(imgs_01, titles, out_path):
    """
    imgs_01: list of (1,3,H,W) tensors in [0,1]
    titles : list[str] same length as imgs_01
    """
    assert len(imgs_01) == len(titles) and len(imgs_01) > 0
    pil_imgs = []
    for t in imgs_01:
        t = t.squeeze(0).clamp(0,1).permute(1,2,0).cpu().numpy()
        arr = (t * 255.0).round().astype(np.uint8)
        pil_imgs.append(Image.fromarray(arr))

    W, H = pil_imgs[0].size
    N = len(pil_imgs)
    title_h = max(32, int(0.08 * H))  # title bar height
    canvas = Image.new("RGB", (W * N, H + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")

    # Try a nicer font if system has it; fall back to default
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=int(title_h * 0.55))
    except Exception:
        font = ImageFont.load_default()

    # Draw per-tile title bars + paste images
    for i, (img, title) in enumerate(zip(pil_imgs, titles)):
        x0 = i * W
        # semi-transparent bar
        draw.rectangle([(x0, 0), (x0 + W, title_h)], fill=(0, 0, 0, 160))
        # centered title
        tw, th = draw.textbbox((0, 0), title, font=font)[2:]
        tx = x0 + (W - tw) // 2
        ty = (title_h - th) // 2
        draw.text((tx, ty), title, font=font, fill=(255, 255, 255))
        # paste image under the bar
        canvas.paste(img, (x0, title_h))

    canvas.save(out_path)

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
    ap.add_argument("--tile", type=int, default=0, help="Enable tiled inference with this tile size (e.g., 512)")
    ap.add_argument("--overlap", type=int, default=32, help="Tile overlap (pixels)")
    ap.add_argument("--no-panels", action="store_true", help="Do not save comparison panels")
    ap.add_argument("--max-side", type=int, default=0, help="If >0, downscale so max(H,W)<=max-side")
    ap.add_argument("--max-pixels", type=int, default=0, help="If >0, downscale so H*W<=max-pixels")
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
    print(f"Device: {device}")

    # Build model shell and load trained ckpt
    net = build_mambairv2_colorizer(
        upscale=int(cfg["model"]["upscale"]),
        in_chans=int(cfg["model"]["in_chans"]),
        img_size=int(cfg["model"]["img_size"]),
        img_range=float(cfg["model"]["img_range"]),
        embed_dim=cfg["model"]["embed_dim"],
        d_state=int(cfg["model"]["d_state"]),
        depths=tuple(cfg["model"]["depths"]),
        num_heads=tuple(cfg["model"]["num_heads"]),
        window_size=int(cfg["model"]["window_size"]),
        inner_rank=int(cfg["model"]["inner_rank"]),
        num_tokens=int(cfg["model"]["num_tokens"]),
        convffn_kernel_size=int(cfg["model"]["convffn_kernel_size"]),
        mlp_ratio=float(cfg["model"]["mlp_ratio"]),
        pretrained=cfg.get("pretrained"),
        device=device
    ).eval()

    have_gt = gt_root is not None and os.path.isdir(gt_root)
    lpips_list, psnr_list, ssim_list, names = [], [], [], []

    amp_enabled = bool(cfg.get("amp", True)) and device.type == "cuda"
    autocast_ctx = autocast if amp_enabled else nullcontext
    with torch.inference_mode(), autocast_ctx():
        for i, p in enumerate(files, 1):
            name = os.path.basename(p)
            # Load input image (RGB), then convert to 3-ch grayscale for the model
            rgb_pil = Image.open(p).convert("RGB")
            rgb_pil = maybe_downscale_pil(rgb_pil, max_side=args.max_side, max_pixels=args.max_pixels)

            x_in = rgb_pil_to_gray3_tensor(rgb_pil).to(device)  # (1,3,H,W)
            
            # Inference (tiled if requested)
            if args.tile and args.tile > 0:
                pred_rgb = forward_tiled(
                    net, x_in, tile=args.tile, overlap=args.overlap,
                    pad_mult=int(cfg["model"].get("window_size", 8))
            )
            else:
                pred_rgb = net(x_in)

            pred_rgb = pred_rgb.clamp(0, 1)   # ensures [0,1] smoothly

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

            
            if not args.no_panels:
                imgs_cpu = [t.cpu() for t in imgs]  # ensure CPU
                save_panel_with_titles(imgs_cpu, titles, panel_dir / name)

            torch.cuda.empty_cache()

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