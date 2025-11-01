import torch
from torch.nn import Module
import torch.nn.functional as F

from pathlib import Path
import os, csv, time
from typing import Optional

from math import log10
import lpips

import matplotlib
# Prefer interactive backend if available; otherwise fall back to Agg
if os.environ.get("DISPLAY", "") == "" and os.environ.get("MPLBACKEND", "") == "":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ------------------- Charbonnier loss ------------------------
def charbonnier(x, y, eps=1e-3):
        return torch.mean(torch.sqrt((x - y)**2 + eps**2))

# ----------------- Checkpoint IO Utils ------------------
def save_ckpt(path: Path, model: Module, opt, scaler, step, best_total, ema=None):
    """Save model, optimizer, scaler, and EMA state to a checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "best_total": best_total,
        "ema": (ema.shadow if ema is not None else None),
    }, path)

def load_ckpt(path, model: Module, opt=None, scaler=None):
    """Load model, optimizer, scaler, and EMA state from a checkpoint."""
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"], strict=False)
    if opt is not None and ck.get("optimizer"):
        opt.load_state_dict(ck["optimizer"])
    if scaler is not None and ck.get("scaler") is not None:
        scaler.load_state_dict(ck["scaler"])
    step = ck.get("step", 0)
    best_total = ck.get("best_total", 1e9)
    ema_sd = ck.get("ema", None)
    return step, best_total, ema_sd

# ----------------- Metrics Utils --------------------
_lpips = lpips.LPIPS(net='vgg').eval()
for p in _lpips.parameters(): p.requires_grad = False

def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return 99.0 if mse == 0 else 10*log10(1.0/mse)

def _lpips_device():
    # current device of the LPIPS module
    try:
        return next(_lpips.parameters()).device
    except StopIteration:
        return torch.device('cpu')

def lpips_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a, b expected in [0,1]. We move inputs to the LPIPS module's device
    and cast to float32 to avoid AMP half-precision issues.
    """
    dev = _lpips_device()
    a = (a * 2 - 1).to(dev, dtype=torch.float32, non_blocking=True)
    b = (b * 2 - 1).to(dev, dtype=torch.float32, non_blocking=True)
    return _lpips(a, b).mean()

# --- YUV chroma-aware loss ---
def rgb_to_yuv(x):
    r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
    y = 0.299*r + 0.587*g + 0.114*b
    u = 0.492*(b - y)
    v = 0.877*(r - y)
    return y, u, v

def yuv_to_rgb(y: torch.Tensor, u: torch.Tensor, v: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    """
    Inverse of rgb_to_yuv used in this repo.
    Inputs: y,u,v in [B,1,H,W]. Returns RGB in [B,3,H,W].
    """
    # Invert the linear mapping used in rgb_to_yuv
    r = v * (1.0 / 0.877) + y
    b = u * (1.0 / 0.492) + y
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    x = torch.cat([r, g, b], dim=1)
    return x.clamp(0, 1) if clamp else x

def yuv3_to_rgb(x_yuv: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    """
    Convenience: x_yuv is [B,3,H,W] with channels [Y,U,V].
    """
    y, u, v = x_yuv[:, 0:1], x_yuv[:, 1:2], x_yuv[:, 2:3]
    return yuv_to_rgb(y, u, v, clamp=clamp)

# --- dynamic chroma weighting ---
def dynamic_chroma_weighting(epoch: int, total_epochs: int, base_lambda_uv: float) -> float:
    """
    Returns a decayed lambda_uv in [0.4*base, 1.0*base] across training.
    - epoch: 1-based current epoch
    - total_epochs: total number of epochs
    - base_lambda_uv: the lambda_uv from the config
    """
    if total_epochs <= 0:
        return float(base_lambda_uv)
    # fades from 1.0 → 0.4 as epoch goes 1 → total_epochs
    progress = max(0.0, min(1.0, (epoch - 1) / max(1, total_epochs - 1)))
    decay = max(0.4, 1.0 - 0.6 * progress)
    return float(base_lambda_uv) * float(decay)

# --- Chroma-weighted UV loss and saturation prior ---
def chroma_weighted_uv_loss(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor, wmin: float = 0.5, wmax: float = 2.0) -> torch.Tensor:
    """
    Weight UV error by ground-truth chroma magnitude so colorful regions matter more.
    pred_rgb, gt_rgb: [B,3,H,W] in [0,1].
    """
    _, u1, v1 = rgb_to_yuv(pred_rgb)
    _, u2, v2 = rgb_to_yuv(gt_rgb)
    chroma_gt = torch.sqrt(u2**2 + v2**2) + 1e-6
    mean_per_img = chroma_gt.mean(dim=[1,2,3], keepdim=True).clamp_min(1e-6)
    w = (chroma_gt / mean_per_img).clamp(wmin, wmax)
    return ((u1 - u2).abs() + (v1 - v2).abs()).mul(w).mean()

def saturation_prior(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    """
    Small penalty to discourage vanishing chroma where GT is colorful.
    Only active where ||UV_gt|| > tau.
    """
    _, up, vp = rgb_to_yuv(pred_rgb)
    _, ug, vg = rgb_to_yuv(gt_rgb)
    chroma_pred = torch.sqrt(up**2 + vp**2)
    chroma_gt   = torch.sqrt(ug**2 + vg**2)
    mask = (chroma_gt > tau).float()
    return (torch.relu(tau - chroma_pred) * mask).mean()

# ----------- Image Panel Drawing Helper------------
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

# --------- Locking Middle Layer Helper -----------
def freeze_all_but_last(net, last_k_blocks=0, extra_modules=("conv_first","conv_after_body","conv_last")):
    for p in net.parameters():
        p.requires_grad = False

    # always train these light heads
    for name in extra_modules:
        if hasattr(net, name):
            for p in getattr(net, name).parameters():
                p.requires_grad = True

    # optionally train last K high-level blocks (if model exposes them as a list/ModuleList)
    if hasattr(net, "layers") and isinstance(net.layers, torch.nn.ModuleList) and last_k_blocks > 0:
        for m in net.layers[-last_k_blocks:]:
            for p in m.parameters():
                p.requires_grad = True

    # sanity: print trainable parameter count
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Trainable params: {n_train:,}")

# --------- Statistics tracker Utils -------------
class StatTracker:
    """
    Lightweight training/validation tracker with live plots via matplotlib draw().
    - Records: step, l1, lpips, total_loss, lr, val_lpips
    - Writes: <out_dir>/logs/train_log.csv
    - Plots:  2 panels -> Train (losses/LR) and Validation (LPIPS)
    - Also writes epoch timing to <out_dir>/logs/epochs_log.csv and total timing to runtime.txt
    """

    def __init__(
        self,
        out_dir: Path,
        redraw_every: int = 50,
        smoothing: float = 0.0,  # EMA smoothing for plotted loss curves (0 = off)
        is_main: bool = True,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "logs" / "train_log.csv"
        self.epoch_csv_path = self.out_dir / "logs" / "epochs_log.csv"
        self.runtime_path = self.out_dir / "runtime.txt"
        self.svg_path = self.out_dir / "plots.svg"
        self.redraw_every = max(1, int(redraw_every))
        self.is_main = bool(is_main)
        self.smoothing = float(smoothing)

        # Buffers
        self.steps = []
        self.l1 = []
        self.lp = []
        self.uv = []
        self.tot = []
        self.lr = []
        self.luv = []   # effective lambda_uv(t) per train step

        self.val_steps = []
        self.val_lp = []
        self.val_l1 = []
        self.val_uv = []
        self.val_tot = []

        # Smoothed
        self._ema_l1 = None
        self._ema_lp = None
        self._ema_tot = None

        # Timing
        self._t0 = time.time()
        self._run_start_wall = None
        self._last_redraw_step = -10**9

        # CSV init
        if self.is_main:
            (self.out_dir / "logs").mkdir(parents=True, exist_ok=True)
            if not self.csv_path.exists():
                with open(self.csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["step", "loss_l1", "loss_lpips", "loss_uv", "loss_total", "lr", "lambda_uv_eff", "val_l1", "val_lpips", "val_uv", "val_total"])

        # Plot init (only on main)
        self._headless = matplotlib.get_backend().lower() == "agg"
        self.fig = None
        self.ax_train = None
        self.ax_val = None
        self.lines = {}  # name -> Line2D

        if self.is_main:
            self._init_plot()

    # --------------- public API ---------------

    def start_run(self):
        """Mark run start and initialize epoch CSV."""
        if not self.is_main:
            return
        self._run_start_wall = time.time()
        if not self.epoch_csv_path.exists():
            with open(self.epoch_csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "duration_sec", "steps", "steps_per_sec", "total_elapsed_sec"])

    def end_run(self, total_duration_sec: float):
        """Write total runtime."""
        if not self.is_main:
            return
        try:
            with open(self.runtime_path, "w") as f:
                f.write(f"total_duration_sec,{total_duration_sec:.6f}\n")
                f.write(f"total_duration_min,{total_duration_sec/60.0:.6f}\n")
        except Exception:
            pass

    def log_epoch(self, epoch: int, duration_sec: float, steps: int, steps_per_sec: float):
        """Append per-epoch timing to epoch CSV."""
        if not self.is_main:
            return
        elapsed = (time.time() - self._run_start_wall) if self._run_start_wall else duration_sec
        with open(self.epoch_csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([int(epoch), float(duration_sec), int(steps), float(steps_per_sec), float(elapsed)])

    def log_train(self, step: int, loss_l1: float, loss_lp: float, loss_uv: Optional[float], total_loss: Optional[float], lr: float, lambda_uv_eff: Optional[float] = None):
        """Record a training step. total_loss can be None; we’ll compute loss_l1+loss_lp if so."""
        if total_loss is None:
            total_loss = float(loss_l1) + float(loss_lp)
        if loss_uv is None:
            loss_uv = 0.0
        if lambda_uv_eff is None:
            lambda_uv_eff = 0.0

        self.steps.append(int(step))
        self.l1.append(float(loss_l1))
        self.lp.append(float(loss_lp))
        self.tot.append(float(total_loss))
        self.lr.append(float(lr))
        self.uv.append(float(loss_uv))
        self.luv.append(float(lambda_uv_eff))

        # EMA smoothing (for display only)
        if self.smoothing > 0:
            self._ema_l1 = self._ema(loss_l1, self._ema_l1, self.smoothing)
            self._ema_lp = self._ema(loss_lp, self._ema_lp, self.smoothing)
            self._ema_tot = self._ema(total_loss, self._ema_tot, self.smoothing)

        # Append row to CSV
        if self.is_main:
            self._append_csv(step, loss_l1, loss_lp, loss_uv, total_loss, lr, lambda_uv_eff, val_l1=None, val_lp=None, val_uv=None, val_total=None)

        # Redraw if needed
        if self.is_main and (step - self._last_redraw_step) >= self.redraw_every:
            self.redraw()

    def log_val(self, step: int, loss_l1: float, loss_lp: float, loss_uv: float, loss_total: float):
        """Record validation metrics (L1, LPIPS, UV, Total) and update plot."""
        step = int(step)
        self.val_steps.append(step)
        self.val_l1.append(float(loss_l1))
        self.val_lp.append(float(loss_lp))
        self.val_uv.append(float(loss_uv))
        self.val_tot.append(float(loss_total))

        if self.is_main:
            self._append_csv(step,
                            loss_l1=None, loss_lp=None, loss_uv=None, total_loss=None, lr=None, lambda_uv_eff=None,
                            val_l1=loss_l1, val_lp=loss_lp, val_uv=loss_uv, val_total=loss_total)
            self.redraw(force=True)

    def redraw(self, force: bool = False):
        """Update live plot using draw()/pause(). In headless, save a SVG instead."""
        if not self.is_main or self.fig is None:
            return
        if not force and len(self.steps) > 0 and (self.steps[-1] - self._last_redraw_step) < self.redraw_every:
            return

        self._update_train_axes()
        self._update_val_axes()

        self.fig.tight_layout()
        if self._headless:
            self.fig.savefig(self.svg_path, dpi=150)
        else:
            plt.draw()
            self.ax_train.legend(loc="upper right")
            self.ax_val.legend(loc="upper right")
            plt.pause(0.001)

        self._last_redraw_step = self.steps[-1] if self.steps else self._last_redraw_step

    def save_fig(self, path: Optional[Path] = None):
        if not self.is_main or self.fig is None:
            return
        p = Path(path) if path is not None else self.svg_path
        self.fig.savefig(p, dpi=150)

    def close(self):
        if self.fig is not None and not self._headless:
            try:
                plt.close(self.fig)
            except Exception:
                pass

    # --------------- internals ---------------

    def _init_plot(self):
        plt.ion()
        self.fig = plt.figure(figsize=(10, 5))
        self.ax_train = self.fig.add_subplot(1, 2, 1)
        self.ax_val = self.fig.add_subplot(1, 2, 2)

        # ----- Training panel -----
        (l1_line,) = self.ax_train.plot([], [], label="L1")
        (lp_line,) = self.ax_train.plot([], [], label="LPIPS")
        (uv_line,) = self.ax_train.plot([], [], label="UV")
        (tot_line,) = self.ax_train.plot([], [], label="Total")
        (lr_line,) = self.ax_train.plot([], [], label="LR (scaled)")
        (luv_line,) = self.ax_train.plot([], [], label="λ_uv (scaled)")

        self.ax_train.set_title("Training losses / LR / λuv")
        self.ax_train.set_xlabel("Step")
        self.ax_train.set_ylabel("Loss")
        self.ax_train.legend(loc="upper right")

        self.lines.update({
            "l1": l1_line,
            "lp": lp_line,
            "uv": uv_line,
            "tot": tot_line,
            "lr": lr_line,
            "luv": luv_line,
        })

        # ----- Validation panel -----
        (vl1_line,)  = self.ax_val.plot([], [], label="Val L1")
        (vlp_line,)  = self.ax_val.plot([], [], label="Val LPIPS")
        (vuv_line,)  = self.ax_val.plot([], [], label="Val UV")
        (vtot_line,) = self.ax_val.plot([], [], label="Val Total")

        self.ax_val.set_title("Validation losses")
        self.ax_val.set_xlabel("Step")
        self.ax_val.set_ylabel("Loss")
        self.ax_val.legend(loc="upper right")

        self.lines.update({
            "val_l1": vl1_line,
            "val_lp": vlp_line,
            "val_uv": vuv_line,
            "val_tot": vtot_line,
        })

        if not self._headless:
            plt.draw()
            plt.pause(0.001)

    def _update_train_axes(self):
        x = self.steps
        if not x:
            return

        l1 = self._series(self.l1, self._ema_l1)
        lp = self._series(self.lp, self._ema_lp)
        tot = self._series(self.tot, self._ema_tot)
        uv = self.uv

        if self.lr:
            lrmax = max(self.lr)
            scale = max(1e-12, lrmax)
            base = max(1e-6, (sum(tot) / len(tot)) if tot else 1.0)
            lr_scaled = [v / scale * base for v in self.lr]
        else:
            lr_scaled = []

        # Scale λ_uv to roughly the same range as losses
        if self.luv:
            luvmax = max(self.luv)
            luv_scale = max(1e-12, luvmax)
            base = max(1e-6, (sum(tot) / len(tot)) if tot else 1.0)
            luv_scaled = [v / luv_scale * base for v in self.luv]
        else:
            luv_scaled = []

        self.lines["l1"].set_data(x, l1)
        self.lines["lp"].set_data(x, lp)
        self.lines["tot"].set_data(x, tot)
        self.lines["lr"].set_data(x, lr_scaled)
        self.lines["uv"].set_data(x, uv)
        self.lines["luv"].set_data(x, luv_scaled)

        xmin, xmax = min(x), max(x)
        self.ax_train.set_xlim(xmin, xmax if xmax > xmin else xmin + 1)

        y_vals = []
        y_vals += l1 if l1 else []
        y_vals += lp if lp else []
        y_vals += tot if tot else []
        y_vals += lr_scaled if lr_scaled else []
        y_vals += luv_scaled if luv_scaled else []
        y_vals += uv if uv else []
        if y_vals:
            ymin, ymax = min(y_vals), max(y_vals)
            pad = 0.05 * (ymax - ymin + 1e-12)
            self.ax_train.set_ylim(ymin - pad, ymax + pad)

        elapsed = time.time() - self._t0
        self.ax_train.set_title(f"Train losses / LR  |  steps={xmax}  |  {elapsed/60.0:.1f} min")

    def _update_val_axes(self):
        xv = self.val_steps
        if not xv:
            return

        y_l1  = self.val_l1
        y_lp  = self.val_lp
        y_uv  = self.val_uv
        y_tot = self.val_tot

        # Set data for each line
        self.lines["val_l1"].set_data(xv, y_l1)
        self.lines["val_lp"].set_data(xv, y_lp)
        self.lines["val_uv"].set_data(xv, y_uv)
        self.lines["val_tot"].set_data(xv, y_tot)

        # X limits
        xmin, xmax = min(xv), max(xv)
        self.ax_val.set_xlim(xmin, xmax if xmax > xmin else xmin + 1)

        # Y limits from all series
        y_all = []
        if y_l1:  y_all += y_l1
        if y_lp:  y_all += y_lp
        if y_uv:  y_all += y_uv
        if y_tot: y_all += y_tot
        if y_all:
            ymin, ymax = min(y_all), max(y_all)
            pad = 0.05 * (ymax - ymin + 1e-12)
            self.ax_val.set_ylim(ymin - pad, ymax + pad)

        best_lp = min(y_lp) if y_lp else float("nan")
        last_tot = y_tot[-1] if y_tot else float("nan")
        self.ax_val.set_title(f"Validation losses  |  best LPIPS={best_lp:.4f}  |  last TOTAL={last_tot:.4f}")

    def _append_csv(self, step, loss_l1, loss_lp, loss_uv, total_loss, lr, lambda_uv_eff, val_l1, val_lp, val_uv, val_total):
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                step,
                loss_l1 if loss_l1 is not None else "",
                loss_lp if loss_lp is not None else "",
                loss_uv if loss_uv is not None else "",
                total_loss if total_loss is not None else "",
                lr if lr is not None else "",
                lambda_uv_eff if lambda_uv_eff is not None else "",
                val_l1 if val_l1 is not None else "",
                val_lp if val_lp is not None else "",
                val_uv if val_uv is not None else "",
                val_total if val_total is not None else "",
            ])

    @staticmethod
    def _ema(x, prev, alpha):
        return x if prev is None else (alpha * prev + (1 - alpha) * x)

    def _series(self, raw_list, ema_value):
        if self.smoothing <= 0 or not raw_list:
            return raw_list
        out = []
        ema = None
        for v in raw_list:
            ema = v if ema is None else (self.smoothing * ema + (1 - self.smoothing) * v)
            out.append(ema)
        return out