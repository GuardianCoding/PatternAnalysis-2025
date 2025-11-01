"""train.py — Two-stage fine-tuning for grayscale→RGB with MambaIRv2.

Overview
--------
This script fine-tunes a pretrained MambaIRv2 restoration backbone to perform
grayscale→RGB colorization using a *two-stage* curriculum:

  • Stage 1 (stability first): freeze the backbone except the heads + last K blocks.
    This preserves most of the pretrained structure and lets the model learn a
    luminance→chrominance mapping without destabilizing earlier features.

  • Stage 2 (expressivity): unfreeze *all* layers for end-to-end refinement.

Losses (weighted in the total objective)
----------------------------------------
  • L1 / Charbonnier on RGB (pixel fidelity).
  • LPIPS (perceptual similarity on a down-scaled side length for speed).
  • UV chroma loss with dynamic/cosine scheduling for λ_UV to gradually relax
    color pressure as structure converges.
  • Saturation prior that gently rewards plausible color saturation.

DDP / Mixed Precision
---------------------
• DistributedDataParallel (NCCL) is supported; launch with torchrun or equivalent.
• AMP (autocast + GradScaler) is enabled/disabled via config ("amp").

Logging / Outputs
-----------------
• Scalar logs and live plots via StatTracker to `outputs/<run_name>/`.
• Periodic panel composites showcasing {Greyscale, Ground Truth, Prediction}.
• Checkpoints: periodic, and best-total snapshot.

Resuming
--------
If `resume` points to a .ckpt, training resumes global step, scaler state, and
(when available) EMA shadow weights. The current best_total is *not* clobbered.

Usage
-----
    python train.py --config configs/config.yml --exp_name <tag>

TIP: Keep crop_size divisible by 16 for stable down/up sampling inside backbones.
"""
import warnings  # Suppress noisy deps to keep logs readable
warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*deprecated.*")
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")
warnings.filterwarnings("ignore", message="Applied workaround for CuDNN issue, install nvrtc.so")

import os, argparse, time, math
from pathlib import Path
import yaml

import torch
import torch.distributed as dist
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
import torch.nn.functional as F

# Project imports: dataloaders, model factory, and utility helpers
from dataset import (
    build_coco_dataloaders,
    sample_pool_indices,
    sample_epoch_indices,
    build_epoch_subset_loader,
)
from modules import build_mambairv2_colorizer
from utils import (
    set_seed,
    lpips_loss,
    _lpips,  # cached LPIPS network module (created in utils)
    rgb_to_yuv,
    dynamic_chroma_weighting,
    yuv_to_rgb,
    chroma_weighted_uv_loss,
    saturation_prior,
    StatTracker,
    save_ckpt,
    load_ckpt,
    save_panel_with_titles,
    freeze_all_but_last,
    charbonnier,
)

# --------- Global kernel hints (speed/memory) ----------
# TF32: allow lower-precision matmul on tensor cores for speed in AMP contexts.
torch.backends.cuda.matmul.allow_tf32 = True
# Matmul precision hint; "high" can be slower. "medium" is a good compromise.
torch.set_float32_matmul_precision("medium")
# cudnn benchmark: enables best algo search for fixed sizes (speeds up training)
torch.backends.cudnn.benchmark = True


# --------------------- utilities ---------------------
class EMA:
    """Exponential Moving Average (EMA) of model weights.

    This keeps a shadow copy of trainable weights updated by:
        shadow = decay * shadow + (1 - decay) * current

    Why EMA?
    --------
    EMA weights often yield smoother, more stable validation metrics than the
    raw (high-variance) training weights, especially for restoration tasks.

    How to use here
    ---------------
    • During training we call `update()` after optimizer steps.
    • During validation we temporarily swap the EMA weights into the model
      (`store` → `copy_to` → run val → `restore`) to evaluate the smoothed model.

    Notes
    -----
    • Only floating-point tensors are tracked.
    • The shadow state_dict mirrors real model keys for easy load/store.
    • When resuming from a checkpoint with EMA, we restore `shadow` directly.

    Parameters
    ----------
    model : nn.Module
        Model whose (float) parameters are to be tracked.
    decay : float
        EMA decay factor in [0,1). Higher = smoother/laggier; 0 disables updates.
    """
    def __init__(self, model, decay: float):
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}     # EMA weights (same keys as model)
        self._backup: dict[str, torch.Tensor] | None = None  # temporary store for swap-in/out
        for k, v in model.state_dict().items():
            # Track only floating-type tensors (params and buffers)
            if torch.is_tensor(v) and v.dtype.is_floating_point:
                t = v.detach().clone()
                t.requires_grad_(False)
                self.shadow[k] = t

    @torch.no_grad()
    def update(self, model):
        """In-place update of EMA shadow from the current model state."""
        if self.decay <= 0:
            return
        msd = model.state_dict()
        for k, s in self.shadow.items():
            v = msd[k]
            # Keep dtype/device in sync before the fused op
            if v.dtype != s.dtype:
                v = v.to(dtype=s.dtype)
            if v.device != s.device:
                v = v.to(device=s.device, non_blocking=True)
            s.mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        """Copy EMA shadow parameters into the model (permanent overwrite)."""
        msd = model.state_dict()
        for k, s in self.shadow.items():
            tgt = msd[k]
            if tgt.device != s.device:
                s = s.to(device=tgt.device, non_blocking=True)
            if tgt.dtype != s.dtype:
                s = s.to(dtype=tgt.dtype)
            tgt.copy_(s)

    @torch.no_grad()
    def store(self, model):
        """Keep a backup of current (non-EMA) weights to restore after validation."""
        self._backup = {}
        for k, s in self.shadow.items():
            self._backup[k] = model.state_dict()[k].detach().clone()

    @torch.no_grad()
    def copy_to(self, model):
        """Swap EMA → model weights (use with store/restore for temporary swap)."""
        self.apply_to(model)

    @torch.no_grad()
    def restore(self, model):
        """Restore the weights saved by `store()` after an EMA validation pass."""
        if self._backup is None:
            return
        msd = model.state_dict()
        for k, v in self._backup.items():
            # Keep copy cheap & safe regardless of dtype/device differences.
            if msd[k].device != v.device:
                v = v.to(device=msd[k].device, non_blocking=True)
            if msd[k].dtype != v.dtype:
                v = v.to(dtype=msd[k].dtype)
            msd[k].copy_(v)
        self._backup = None

    def state_dict(self):
        """Return a CPU-resident (portable) snapshot of the EMA shadow."""
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state):
        """Load an EMA snapshot (e.g., from checkpoint)."""
        self.decay = float(state["decay"])
        self.shadow = {k: v.clone().detach() for k, v in state["shadow"].items()}
        for t in self.shadow.values():
            t.requires_grad_(False)


class WarmupCosine(torch.optim.lr_scheduler._LRScheduler):
    """Linear warm-up followed by cosine decay to a minimum LR.

    Behavior
    --------
    • For the first `warmup_steps`, LR increases linearly from 0 → base_lr.
    • Thereafter, LR follows a cosine schedule from base_lr → min_lr over
      the remaining steps (until `max_steps`).

    Why this schedule?
    ------------------
    Warm-up reduces early instability (esp. with AMP + large batch),
    cosine decay is a simple, strong general-purpose schedule for finetuning.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to schedule.
    base_lr : float
        Peak learning rate after warm-up.
    warmup_steps : int
        Number of warm-up steps (>=1).
    max_steps : int
        Total training steps for the schedule horizon.
    min_lr : float, optional
        Floor LR at the end of the cosine, by default 1e-5.
    last_epoch : int, optional
        Internal PyTorch bookkeeping; ignore for normal use.
    """
    def __init__(self, optimizer, base_lr, warmup_steps, max_steps, min_lr=1e-5, last_epoch=-1):
        self.base_lr   = float(base_lr)
        self.warmup    = max(1, int(warmup_steps))
        self.max_steps = max_steps
        self.min_lr    = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        # Warm-up: linearly scale from 0 → base_lr
        if self.max_steps is None or step <= self.warmup:
            scale = min(1.0, step / self.warmup)
            return [self.base_lr * scale for _ in self.optimizer.param_groups]

        # Cosine phase: smooth decay to min_lr
        t = (step - self.warmup) / (self.max_steps - self.warmup)
        cos = 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, t))))
        return [self.min_lr + (self.base_lr - self.min_lr) * cos for _ in self.optimizer.param_groups]


# --------------------- λ_UV schedules ---------------------
def cosine_decay_lambda_uv(epoch: int, total_epochs: int, start: float, end: float = 0.0) -> float:
    """Cosine-decay λ_UV across epochs (monotone from `start` → `end`).

    Use when:
      • You want strong early color pressure that gradually relaxes as structure
        (L1/LPIPS) converges.

    Returns
    -------
    float
        Effective λ_UV for the current `epoch` (1-indexed).
    """
    if total_epochs <= 1:
        return float(end)
    t = (epoch - 1) / float(total_epochs - 1)
    return float(end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * t)))


def cosine_hold_decay_lambda_uv(
    epoch: int,
    total_epochs: int,
    start: float,
    end: float = 0.0,
    hold_pct: float = 0.6,
) -> float:
    """Hold-then-decay schedule for λ_UV.

    Behavior
    --------
    • Hold λ_UV at `start` for the first `hold_pct` of training (by epoch).
    • Then cosine-decay the remainder down to `end`.

    Why prefer this?
    ----------------
    Useful if early experiments show under-colorization even after a few epochs;
    holding sustains a stronger chroma signal to kickstart color learning.

    Returns
    -------
    float
        Effective λ_UV for the current `epoch` (1-indexed).
    """
    if total_epochs <= 1:
        return float(end)
    hold_e = int(round(max(0.0, min(1.0, hold_pct)) * (total_epochs - 1))) + 1
    if epoch <= hold_e:
        return float(start)
    t = (epoch - hold_e) / float(max(1, total_epochs - hold_e))
    cos = 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, t))))
    return float(end + (start - end) * cos)


# ------------------------- Validation Loop ----------------------------
def validate(
    net: Module,
    val_loader,
    ema: EMA,
    epoch,
    total_epochs,
    tracker: StatTracker,
    global_step: int,
    device,
    cfg,
    is_main: bool,
    use_ddp,
    out_root,
    best_total,
    opt,
    scaler,
):
    """Run a validation pass, optionally evaluating the EMA weights.

    Flow
    ----
    1) Swap in EMA weights (store→copy_to) if EMA is enabled.
    2) Disable grad, run forward on val set, and accumulate the four losses.
    3) Compute the *validation* total loss using the *current* λ_UV schedule
       (so the score reflects the training objective at this epoch).
    4) Restore original non-EMA weights (if they were swapped).
    5) Log scalar averages via `StatTracker`. If the `TOTAL` improves, snapshot.

    Notes
    -----
    • We compute LPIPS directly at full crop size here; if too slow, consider
      down-scaling similarly to training (`lpips_side`).
    • No DDP synchronization is performed here; only rank-0 logs and saves.

    Parameters
    ----------
    net : nn.Module
        The training model (possibly wrapped in DDP).
    val_loader : DataLoader
        Validation iterator yielding (x_gray3, y_rgb).
    ema : EMA | None
        EMA handler; when provided, will be used for evaluation.
    epoch, total_epochs : int
        For schedule evaluation (λ_UV).
    tracker : StatTracker
        Handles CSV + plot updates on rank-0.
    global_step : int
        Current global step for x-axis alignment in plots.
    device : torch.device
        Target device (cuda/cpu).
    cfg : dict
        Configuration (weights, schedules, etc.).
    is_main : bool
        Rank-0 indicator; only rank-0 prints and saves.
    use_ddp : bool
        If True, `net` is DDP-wrapped and we access `.module` when needed.
    out_root : Path
        Run directory root (for checkpoints).
    best_total : float
        Current best validation total loss (lower is better).
    opt, scaler : Optimizer, GradScaler
        Passed through for checkpoint serialization.
    """
    net.eval()
    val_l1 = val_lp = val_uv = val_sat = val_total = 0.0
    n_count = 0

    # Pull weights/schedule knobs once for clarity
    lambda_uv_sched = str(cfg.get("lambda_uv_schedule", "cosine")).lower()
    lambda_uv      = float(cfg.get("lambda_uv", 6.0))
    lambda_uv_min  = float(cfg.get("lambda_uv_min", 3.0))
    hold_pct       = float(cfg.get("lambda_uv_hold_pct", 0.6))
    w_l1           = float(cfg.get("lambda_l1", 0.12))
    w_lp           = float(cfg.get("lambda_lpips", 1.0))
    lam_sat        = float(cfg.get("lambda_sat", 0.05))
    uv_wmin        = float(cfg.get("uv_wmin", 0.5))
    uv_wmax        = float(cfg.get("uv_wmax", 2.0))

    # Swap EMA weights in for evaluation (if configured)
    with torch.no_grad():
        swapped = False
        if ema is not None:
            ema.store(net)
            ema.copy_to(net)
            swapped = True

        for batch in val_loader:
            # B,C,H,W tensors in [0,1]
            x_v, y_v = batch[0], batch[1]
            x_v = x_v.to(device, non_blocking=True)
            y_v = y_v.to(device, non_blocking=True)

            pred_v = net(x_v)

            # Component losses (float scalars)
            l1 = F.l1_loss(pred_v, y_v).item()
            lp = float(lpips_loss(pred_v, y_v).item())
            uv = float(chroma_weighted_uv_loss(pred_v, y_v, wmin=uv_wmin, wmax=uv_wmax).item())
            sat_v = float(saturation_prior(pred_v, y_v, tau=float(cfg.get("sat_tau", 0.05))).item())

            # Compute scheduled λ_UV for this epoch
            if lambda_uv_sched == "cosine_hold":
                lambda_uv_eff_v = cosine_hold_decay_lambda_uv(
                    epoch, total_epochs, lambda_uv, lambda_uv_min, hold_pct=hold_pct
                )
            elif lambda_uv_sched == "cosine":
                lambda_uv_eff_v = cosine_decay_lambda_uv(epoch, total_epochs, lambda_uv, lambda_uv_min)
            else:
                lambda_uv_eff_v = dynamic_chroma_weighting(epoch, total_epochs, lambda_uv)

            # Composite
            total = (w_l1 * l1) + (w_lp * lp) + (lambda_uv_eff_v * uv) + (lam_sat * sat_v)

            # Accumulate
            val_l1   += l1
            val_lp   += lp
            val_uv   += uv
            val_sat  += sat_v
            val_total += total
            n_count  += 1

        # Restore original (non-EMA) weights after evaluation
        if swapped:
            ema.restore(net)

    # Averages
    if n_count > 0:
        avg_l1   = val_l1 / n_count
        avg_lp   = val_lp / n_count
        avg_uv   = val_uv / n_count
        avg_sat  = val_sat / n_count
        avg_total = val_total / n_count

    # Log row + terse console line
    tracker.log_val(
        global_step,
        loss_l1=avg_l1,
        loss_lp=avg_lp,
        loss_uv=avg_uv,
        loss_sat=avg_sat,
        loss_total=avg_total,
    )
    print(
        f"[val] step={global_step} "
        f"L1={avg_l1:.4f} LPIPS={avg_lp:.4f} UV={avg_uv:.4f} SAT={avg_sat:.4f} TOTAL={avg_total:.4f}"
    )

    # Best-so-far checkpoint (TOTAL)
    if (avg_total < best_total) and is_main:
        save_ckpt(
            out_root / "checkpoints" / "best_total.ckpt",
            net.module if use_ddp else net,
            opt,
            scaler,
            global_step,
            avg_total,
            ema,
        )


# ----------------------- Epoch running script --------------------------
def run_training_epochs(
    net: Module,
    opt,
    sched,
    scaler,
    ema,
    base_train_ds,
    eval_loader,
    cfg,
    device,
    out_root,
    tracker: StatTracker,
    is_main,
    use_ddp,
    rank,
    start_epoch,
    end_epoch,
    total_epochs,
    global_step,
    best_total,
    pool_indices,
):
    """Run epochs [start_epoch, end_epoch] inclusive with the standard inner loop.

    What this does
    --------------
    • Builds a per-epoch subset DataLoader from a fixed pool (deterministic), keeping
      the gradient signal fresh while bounding epoch cost on large datasets.
    • Trains with AMP + grad accumulation (for larger effective batch size).
    • Logs per-`log_every` steps; validates/saves per cadence in config.
    • Periodically saves side-by-side panels to visually audit colorization.

    Why subset sampling?
    --------------------
    For massive datasets, using a fixed pool + reshuffled epoch subsets
    maintains diversity over time while ensuring each epoch has predictable cost.

    Returns
    -------
    (global_step, best_total)
        Updated counters after finishing the epoch range.
    """
    # Pull frequently used knobs once
    grad_accum      = max(1, int(cfg.get("grad_accum", 1)))
    save_every      = int(cfg.get("save_every", 2000))
    val_every       = int(cfg.get("val_every", 1000))
    panel_every     = int(cfg.get("panel_every", 0))
    lpips_side      = int(cfg.get("lpips_side", min(int(cfg.get("crop_size", 128)), 192)))
    lambda_uv       = float(cfg.get("lambda_uv", 0.8))
    lambda_uv_min   = float(cfg.get("lambda_uv_min", 0.0))
    lambda_uv_sched = str(cfg.get("lambda_uv_schedule", "dynamic")).lower()
    w_l1            = float(cfg.get("lambda_l1", 1.0))
    w_lp            = float(cfg.get("lambda_lpips", 0.4))
    log_every       = int(cfg.get("log_every", 100))
    use_amp         = bool(cfg.get("amp", True))
    epoch_subset_size = int(cfg.get("epoch_subset_size", 5000))
    pool_seed         = int(cfg.get("train_pool_seed", 1337))

    train_wall_start = time.time()

    for epoch in range(start_epoch, end_epoch + 1):
        # Optional: bias crop selection toward regions with richer chroma in early epochs
        warm = int(cfg.get("chroma_bias_warmup_epochs", 0))
        if hasattr(base_train_ds, "set_bias_active"):
            base_train_ds.set_bias_active(epoch <= warm)

        # Deterministic per-epoch subset from the fixed pool
        epoch_indices = sample_epoch_indices(pool_indices, epoch_subset_size, seed=pool_seed, epoch=epoch)
        train_loader, train_samp = build_epoch_subset_loader(base_train_ds, epoch_indices, cfg, use_ddp, rank)
        if use_ddp and train_samp is not None:
            # Set epoch to reshuffle DDP shard order deterministically
            train_samp.set_epoch(epoch)

        epoch_start = time.time()
        steps_in_epoch = 0

        net.train()
        opt.zero_grad(set_to_none=True)

        for batch in train_loader:
            steps_in_epoch += 1

            # Move mini-batch to device with channels_last to better use TensorCores
            x_in, y_tgt = batch[0], batch[1]
            x_in  = x_in.to(device, non_blocking=True, memory_format=torch.channels_last)
            y_tgt = y_tgt.to(device, non_blocking=True, memory_format=torch.channels_last)

            # --- Chroma "nudge": add tiny UV noise to inputs to discourage a degenerate
            #     "copy grayscale to all channels" solution at the start of training.
            if cfg.get("uv_input_dither", True) and net.training:
                with torch.no_grad():
                    y, u, v = rgb_to_yuv(x_in)
                    std = float(cfg.get("uv_dither_std", 0.01))
                    if std > 0:
                        u = u + std * torch.randn_like(u)
                        v = v + std * torch.randn_like(v)
                        x_in = yuv_to_rgb(y, u, v, clamp=True).contiguous(memory_format=torch.channels_last)

            with autocast(enabled=use_amp):
                # Forward
                pred_rgb = net(x_in)

                # Losses
                loss_l1  = charbonnier(pred_rgb, y_tgt)  # robust L1
                # For LPIPS speed and regularization, compute on a smaller side
                pr_s = F.interpolate(pred_rgb, size=lpips_side, mode="bilinear", align_corners=False)
                gt_s = F.interpolate(y_tgt,     size=lpips_side, mode="bilinear", align_corners=False)
                loss_lp = lpips_loss(pr_s, gt_s)

                # Chroma loss in UV space with pixel-wise weights bounded in [wmin,wmax]
                loss_uv = chroma_weighted_uv_loss(
                    pred_rgb, y_tgt,
                    wmin=float(cfg.get("uv_wmin", 0.5)),
                    wmax=float(cfg.get("uv_wmax", 2.0)),
                )

                # Schedule λ_UV by epoch (cosine, hold-cosine, or dynamic heuristic)
                if lambda_uv_sched == "cosine_hold":
                    lambda_uv_eff = cosine_hold_decay_lambda_uv(
                        epoch, total_epochs, lambda_uv, lambda_uv_min,
                        hold_pct=float(cfg.get("lambda_uv_hold_pct", 0.6)),
                    )
                elif lambda_uv_sched == "cosine":
                    lambda_uv_eff = cosine_decay_lambda_uv(epoch, total_epochs, lambda_uv, lambda_uv_min)
                else:
                    lambda_uv_eff = dynamic_chroma_weighting(epoch, total_epochs, lambda_uv)

                # Mild saturation prior to discourage washed-out predictions
                loss_sat = saturation_prior(pred_rgb, y_tgt, tau=float(cfg.get("sat_tau", 0.05)))
                lam_sat = float(cfg.get("lambda_sat", 0.05))

                # Composite (deferred division for grad accumulation)
                loss = (
                    w_l1 * loss_l1 +
                    w_lp * loss_lp +
                    lambda_uv_eff * loss_uv +
                    lam_sat * loss_sat
                ) / grad_accum

            # Backward on scaled loss (AMP-safe)
            scaler.scale(loss).backward()
            global_step += 1

            # Optimizer step every `grad_accum` micro-batches
            if global_step % grad_accum == 0:
                prev = opt._step_count  # detect if an actual opt step occurs
                scaler.step(opt)
                scaler.update()
                # Step LR scheduler *once per optimizer step*
                if opt._step_count > prev:
                    sched.step()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(net.module if use_ddp else net)

            # Periodic scalar logging (rank-0 only)
            if is_main and global_step % log_every == 0:
                current_lr = opt.param_groups[0]["lr"]
                total_now = (
                    w_l1 * loss_l1 +
                    w_lp * loss_lp +
                    lambda_uv_eff * loss_uv +
                    lam_sat * loss_sat
                ).item()
                tracker.log_train(
                    global_step,
                    loss_l1.item(),
                    loss_lp.item(),
                    loss_uv.item(),
                    float(loss_sat.item()),
                    total_now,
                    current_lr,
                    lambda_uv_eff=float(lambda_uv_eff),
                )
                print(
                    f"[{epoch}] step={global_step} "
                    f"l1={loss_l1.item():.4f} lp={loss_lp.item():.4f} uv={loss_uv.item():.4f} "
                    f"lambda_uv={lambda_uv_eff:.3f} lsat={loss_sat:.4f} "
                    f"loss_total={total_now:.4f} lr={current_lr:.2e}"
                )

            # Visual panel (helps catch "still gray" or "weird hues" quickly)
            if is_main and (panel_every > 0) and (global_step % panel_every == 0):
                with torch.no_grad():
                    x0  = x_in[0:1]
                    y0  = y_tgt[0:1].clamp(0, 1)
                    p0  = pred_rgb[0:1].clamp(0, 1)
                    y_lum, _, _ = rgb_to_yuv(x0)
                    gs3 = y_lum.repeat(1, 3, 1, 1).clamp(0, 1)
                    panel_dir = out_root / "panels"
                    panel_dir.mkdir(parents=True, exist_ok=True)
                    panel_path = panel_dir / f"step_{global_step:07d}.jpg"
                    save_panel_with_titles([gs3, y0, p0], ["Greyscale", "Ground truth", "Model Prediction"], panel_path)

            # Periodic validation & checkpointing (rank-0)
            if is_main and val_every > 0 and global_step % val_every == 0:
                validate(
                    net=net,
                    val_loader=eval_loader,
                    ema=ema,
                    epoch=epoch,
                    total_epochs=total_epochs,
                    tracker=tracker,
                    global_step=global_step,
                    device=device,
                    cfg=cfg,
                    is_main=is_main,
                    use_ddp=use_ddp,
                    out_root=out_root,
                    opt=opt,
                    scaler=scaler,
                    best_total=best_total,
                )

            if is_main and save_every > 0 and global_step % save_every == 0:
                save_ckpt(
                    out_root / "checkpoints" / f"step_{global_step}.ckpt",
                    net.module if use_ddp else net,
                    opt,
                    scaler,
                    global_step,
                    best_total,
                    ema,
                )

            # Housekeeping: keep the VRAM watermark lower on long runs
            if global_step % 25 == 0:
                torch.cuda.empty_cache()

        # Epoch summary (throughput helps catch data stalls)
        if is_main:
            epoch_dur = time.time() - epoch_start
            steps_per_sec = steps_in_epoch / max(epoch_dur, 1e-9)
            tracker.log_epoch(
                epoch=epoch,
                duration_sec=epoch_dur,
                steps=steps_in_epoch,
                steps_per_sec=steps_per_sec,
            )
            print(f"[epoch {epoch}] duration={epoch_dur:.2f}s  steps={steps_in_epoch}  {steps_per_sec:.2f} steps/s")

    if is_main:
        total_dur = time.time() - train_wall_start
        print(f"[stage done] wall_time={total_dur/60.0:.2f} min  ({total_dur:.2f}s)")

    return global_step, best_total


# --------------------- main ---------------------
def main():
    """Entry point: read config, set up environment, run Stage 1 then Stage 2.

    CLI
    ---
    --config     : YAML config path (required).
    --train_root : (optional) override root for training images.
    --val_root   : (optional) override root for validation images.
    --ann_root   : (optional) override root for COCO annotations.
    --pretrained : (optional) checkpoint to initialize backbone weights.
    --resume     : (optional) training checkpoint to resume from.
    --exp_name   : tag used to name the run directory.

    Run structure
    -------------
    outputs/
      <exp_name>_<timestamp>/
        checkpoints/
        panels/
        plots/
        metrics.csv / logs.csv (via StatTracker)
        config_merged.yaml
        run_meta.txt

    Notes
    -----
    • DDP: this function initializes the NCCL process group when WORLD_SIZE>1.
    • We symlink `<exp_name>_latest` to the current run for quick access.
    • `StatTracker` will auto-redraw plots every `plot_redraw_every` steps.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train_root", type=str, default=None)
    ap.add_argument("--val_root",   type=str, default=None)
    ap.add_argument("--ann_root",   type=str, default=None)
    ap.add_argument("--pretrained", type=str, default=None)
    ap.add_argument("--resume",     type=str, default=None)
    ap.add_argument("--exp_name",   type=str, default="exp")
    args = ap.parse_args()

    # Merge CLI overrides into the config dict to keep a single source of truth.
    cfg = yaml.safe_load(open(args.config))
    for k in ("train_root", "val_root", "pretrained", "resume", "ann_root"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    # --------- DDP init (if requested via launcher) ----------
    world = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world > 1
    if use_ddp:
        # torchrun sets LOCAL_RANK and initializes env vars; we set CUDA device accordingly.
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    # Device & seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    set_seed(int(cfg.get("seed", 1337)))

    # Allocate the LPIPS network on device once (utils caches the module)
    _lpips.to(device)

    # --------- Run directory & metadata ----------
    exp_base = Path(cfg.get("out_dir", "outputs"))
    exp_name = args.exp_name or cfg.get("exp_defaults", "exp")
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{exp_name}_{ts}"
    out_root: Path = exp_base / run_name
    ckpt_dir: Path = out_root / "checkpoints"

    is_main = (not use_ddp) or dist.get_rank() == 0
    if is_main:
        out_root.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Keep a copy of the resolved config for provenance
        yaml.safe_dump(cfg, open(out_root / "config_merged.yaml", "w"))
        # Human-readable run metadata for quick inspection
        with open(out_root / "run_meta.txt", "w") as f:
            cmd = " ".join(os.sys.argv)
            f.write(f"run_name: {run_name}\n")
            f.write(f"timestamp: {ts}\n")
            f.write(f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}\n")
            f.write(f"cmd: {cmd}\n")
        # Convenience: symlink <exp_name>_latest → this run
        try:
            latest_link = exp_base / f"{exp_name}_latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(out_root.name)
        except Exception:
            # Some filesystems (e.g., Windows) or envs may not support symlinks
            with open(exp_base / f"{exp_name}_latest.txt", "w") as f:
                f.write(str(out_root.resolve()))

    # --------- Tracker / plots ----------
    tracker = StatTracker(
        out_dir=out_root,
        redraw_every=int(cfg.get("plot_redraw_every", 50)),
        smoothing=float(cfg.get("plot_smoothing", 0.0)),
        is_main=is_main,
    )
    if is_main:
        tracker.start_run()

    # --------- Build data loaders ----------
    train_loader, eval_loader, train_samp, eval_samp = build_coco_dataloaders(
        cfg, use_ddp=use_ddp, rank=(dist.get_rank() if use_ddp else 0),
    )
    base_train_ds = train_loader.dataset

    # Fixed pool for epoch subsets (reproducible diversity)
    train_pool_size = int(cfg.get("train_pool_size", 10000))
    train_pool_seed = int(cfg.get("train_pool_seed", 1337))
    pool_indices = sample_pool_indices(len(base_train_ds), train_pool_size, train_pool_seed)
    if is_main:
        print(f"Pool prepared: using {len(pool_indices)} images out of {len(base_train_ds)}")

    # --------- Build model ----------
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
        device=device,
    )
    if use_ddp:
        # No unused params expected; keeps DDP comm efficient.
        net = DDP(net, device_ids=[torch.cuda.current_device()], find_unused_parameters=False)

    # Reset the runtime VRAM watermark for clean diagnostics
    torch.cuda.reset_peak_memory_stats()

    # AMP scaler (safe on CPU as a no-op when CUDA not available)
    scaler = GradScaler(enabled=bool(cfg.get("amp", True)))

    # Optional EMA wrapper
    ema = None
    if float(cfg.get("ema_decay", 0.0)) > 0:
        ema = EMA(net.module if use_ddp else net, decay=float(cfg["ema_decay"]))

    # --------------------- resume ---------------------
    step, best_total, ema_sd = 0, 1e9, None
    if cfg.get("resume"):
        if is_main:
            print(f"[resume] loading {cfg['resume']}")
        _model = net.module if use_ddp else net
        # Dummy optimizer to satisfy checkpoint schema; we don't reuse its state.
        opt_dummy = AdamW(_model.parameters(), lr=1e-6)
        step, best_total, ema_sd = load_ckpt(cfg["resume"], _model, opt_dummy, scaler)
        if ema_sd is not None and ema is not None:
            # Restore EMA shadow directly
            ema.shadow = ema_sd
        if is_main:
            print(f"[resume] step={step} best_total_loss={best_total:.4f}")

    # --------------------- TWO-STAGE TRAINING ---------------------
    total_epochs     = int(cfg.get("epochs", 10))
    stage1_epochs    = int(cfg.get("stage1_epochs", 3))
    last_k_blocks    = int(cfg.get("stage1_last_k_blocks", 2))
    stage1_lr        = float(cfg.get("stage1_lr", 3e-4))
    stage2_lr        = float(cfg.get("lr", 5e-5))
    weight_decay     = float(cfg.get("weight_decay", 1e-4))
    min_lr           = float(cfg.get("min_lr", 1e-5))
    warmup_steps     = int(cfg.get("warmup_steps", 500))

    rank = (dist.get_rank() if use_ddp else 0)

    # ---- Stage 1: partial unfreeze (heads + last K blocks) ----
    if stage1_epochs > 0:
        print(f"\n=== Stage 1: freeze all but heads + last {last_k_blocks} blocks for {stage1_epochs} epochs ===")
        freeze_all_but_last(net.module if use_ddp else net, last_k_blocks=last_k_blocks)

        opt = AdamW(
            filter(lambda p: p.requires_grad, (net.parameters() if not use_ddp else net.module.parameters())),
            lr=stage1_lr,
            weight_decay=weight_decay,
        )
        total_steps_s1 = (
            stage1_epochs
            * max(1, int(cfg.get("epoch_subset_size", 5000)))
            // max(1, int(cfg.get("batch_size", 10)))
        )
        sched = WarmupCosine(opt, base_lr=stage1_lr, warmup_steps=warmup_steps, max_steps=total_steps_s1, min_lr=min_lr)

        step, best_total = run_training_epochs(
            net=net,
            opt=opt,
            sched=sched,
            scaler=scaler,
            ema=ema,
            base_train_ds=base_train_ds,
            eval_loader=eval_loader,
            cfg=cfg,
            device=device,
            out_root=out_root,
            tracker=tracker,
            is_main=is_main,
            use_ddp=use_ddp,
            rank=rank,
            start_epoch=1,
            end_epoch=stage1_epochs,
            total_epochs=total_epochs,
            global_step=step,
            best_total=best_total,
            pool_indices=pool_indices,
        )

    # ---- Stage 2: full unfreeze ----
    if total_epochs > stage1_epochs:
        print(f"\n=== Stage 2: unfreeze ALL layers for remaining {total_epochs - stage1_epochs} epochs ===")
        for p in (net.module if use_ddp else net).parameters():
            p.requires_grad = True

        opt = AdamW(
            (net.parameters() if not use_ddp else net.module.parameters()),
            lr=stage2_lr,
            weight_decay=weight_decay,
        )
        total_steps_s2 = (
            (total_epochs - stage1_epochs)
            * max(1, int(cfg.get("epoch_subset_size", 5000)))
            // max(1, int(cfg.get("batch_size", 10)))
        )
        sched = WarmupCosine(opt, base_lr=stage2_lr, warmup_steps=warmup_steps, max_steps=total_steps_s2, min_lr=min_lr)

        step, best_total = run_training_epochs(
            net=net,
            opt=opt,
            sched=sched,
            scaler=scaler,
            ema=ema,
            base_train_ds=base_train_ds,
            eval_loader=eval_loader,
            cfg=cfg,
            device=device,
            out_root=out_root,
            tracker=tracker,
            is_main=is_main,
            use_ddp=use_ddp,
            rank=rank,
            start_epoch=stage1_epochs + 1,
            end_epoch=total_epochs,
            total_epochs=total_epochs,
            global_step=step,
            best_total=best_total,
            pool_indices=pool_indices,
        )

    # Save plots and close file handles
    if is_main:
        tracker.save_fig()
        tracker.close()

    # Clean DDP
    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()