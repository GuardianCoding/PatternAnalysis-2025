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

from dataset import build_coco_dataloaders, sample_pool_indices, sample_epoch_indices, build_epoch_subset_loader
from modules import build_mambairv2_colorizer
from utils import set_seed, lpips_loss, _lpips, rgb_to_yuv, dynamic_chroma_weighting, yuv_to_rgb, chroma_weighted_uv_loss, saturation_prior
from utils import StatTracker
from utils import save_ckpt, load_ckpt
from utils import save_panel_with_titles
from utils import freeze_all_but_last, charbonnier

# Memory savings
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

# --------------------- utilities ---------------------
class EMA:
    def __init__(self, model, decay: float):
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] | None = None  # for swap-in/out
        for k, v in model.state_dict().items():
            if torch.is_tensor(v) and v.dtype.is_floating_point:
                t = v.detach().clone()
                t.requires_grad_(False)
                self.shadow[k] = t

    @torch.no_grad()
    def update(self, model):
        if self.decay <= 0:
            return
        msd = model.state_dict()
        for k, s in self.shadow.items():
            v = msd[k]
            if v.dtype != s.dtype:
                v = v.to(dtype=s.dtype)
            if v.device != s.device:
                v = v.to(device=s.device, non_blocking=True)
            s.mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    # ---- Apply EMA weights permanently (matches your call site) ----
    @torch.no_grad()
    def apply_to(self, model):
        msd = model.state_dict()
        for k, s in self.shadow.items():
            tgt = msd[k]
            if tgt.device != s.device:
                s = s.to(device=tgt.device, non_blocking=True)
            if tgt.dtype != s.dtype:
                s = s.to(dtype=tgt.dtype)
            tgt.copy_(s)

    # ---- Optional: non-destructive swap for eval ----
    @torch.no_grad()
    def store(self, model):
        """Save current (non-EMA) float weights to restore later."""
        self._backup = {}
        for k, s in self.shadow.items():
            self._backup[k] = model.state_dict()[k].detach().clone()

    @torch.no_grad()
    def copy_to(self, model):
        """Copy EMA -> model (like apply_to but intended for swap)."""
        self.apply_to(model)

    @torch.no_grad()
    def restore(self, model):
        """Restore the weights saved by store()."""
        if self._backup is None:
            return
        msd = model.state_dict()
        for k, v in self._backup.items():
            if msd[k].device != v.device:
                v = v.to(device=msd[k].device, non_blocking=True)
            if msd[k].dtype != v.dtype:
                v = v.to(dtype=msd[k].dtype)
            msd[k].copy_(v)
        self._backup = None

    # ---- (Nice-to-have) checkpointing support for EMA shadow ----
    def state_dict(self):
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state):
        self.decay = float(state["decay"])
        self.shadow = {k: v.clone().detach() for k, v in state["shadow"].items()}
        for t in self.shadow.values():
            t.requires_grad_(False)

class WarmupCosine(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, base_lr, warmup_steps, max_steps, min_lr=1e-5, last_epoch=-1):
        self.base_lr   = float(base_lr)
        self.warmup    = max(1, int(warmup_steps))
        self.max_steps = max_steps
        self.min_lr    = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        # linear warm-up to base_lr
        if self.max_steps is None or step <= self.warmup:
            scale = min(1.0, step / self.warmup)
            return [self.base_lr * scale for _ in self.optimizer.param_groups]
        # cosine decay to min_lr
        t = (step - self.warmup) / (self.max_steps - self.warmup)
        cos = 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, t))))
        return [self.min_lr + (self.base_lr - self.min_lr) * cos for _ in self.optimizer.param_groups]

# --------------------- lambda_uv cosine schedule ---------------------
def cosine_decay_lambda_uv(epoch: int, total_epochs: int, start: float, end: float = 0.0) -> float:
    """
    Cosine decay from `start` at epoch 1 to `end` at epoch `total_epochs`.
    Smooth and monotonic: w_e = end + (start - end) * 0.5 * (1 + cos(pi * t)),
    with t in [0,1] mapping 1..total_epochs -> 0..1.
    """
    if total_epochs <= 1:
        return float(end)
    t = (epoch - 1) / float(total_epochs - 1)
    return float(end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * t)))

def cosine_hold_decay_lambda_uv(epoch: int, total_epochs: int, start: float, end: float = 0.0, hold_pct: float = 0.6) -> float:
    """
    Hold lambda at `start` for first `hold_pct` of epochs, then cosine-decay to `end`.
    """
    if total_epochs <= 1:
        return float(end)
    hold_e = int(round(max(0.0, min(1.0, hold_pct)) * (total_epochs - 1))) + 1
    if epoch <= hold_e:
        return float(start)
    t = (epoch - hold_e) / float(max(1, total_epochs - hold_e))
    cos = 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, t))))
    return float(end + (start - end) * cos)

# ----------------------- Epoch running script --------------------------
def run_training_epochs(
    net: Module, opt, sched, scaler, ema,
    base_train_ds, eval_loader, cfg, device, out_root, tracker: StatTracker,
    is_main, use_ddp, rank,
    start_epoch, end_epoch, total_epochs,
    global_step, best_total, pool_indices
):
    """Runs epochs [start_epoch, end_epoch] inclusive with original inner loop."""
    # pull frequently used cfg knobs once
    grad_accum   = max(1, int(cfg.get("grad_accum", 1)))
    save_every   = int(cfg.get("save_every", 2000))
    val_every    = int(cfg.get("val_every", 1000))
    panel_every  = int(cfg.get("panel_every", 0))
    lpips_side   = int(cfg.get("lpips_side", min(int(cfg.get("crop_size", 128)), 192)))
    lambda_uv    = float(cfg.get("lambda_uv", 0.8))
    lambda_uv_min = float(cfg.get("lambda_uv_min", 0.0))
    lambda_uv_sched = str(cfg.get("lambda_uv_schedule", "dynamic")).lower()
    w_l1         = float(cfg.get("lambda_l1", 1.0))
    w_lp         = float(cfg.get("lambda_lpips", 0.4))
    log_every    = int(cfg.get("log_every", 100))
    use_amp      = bool(cfg.get("amp", True))
    epoch_subset_size = int(cfg.get("epoch_subset_size", 5000))
    pool_seed         = int(cfg.get("train_pool_seed", 1337))

    train_wall_start = time.time()

    for epoch in range(start_epoch, end_epoch + 1):
        # --- build per-epoch dataloader over a fresh random subset from the fixed pool
        epoch_indices = sample_epoch_indices(pool_indices, epoch_subset_size, seed=pool_seed, epoch=epoch)
        train_loader, train_samp = build_epoch_subset_loader(base_train_ds, epoch_indices, cfg, use_ddp, rank)
        if use_ddp and train_samp is not None:
            train_samp.set_epoch(epoch)

        # ---- epoch timing start ----
        epoch_start = time.time()
        steps_in_epoch = 0

        net.train()
        opt.zero_grad(set_to_none=True)

        for x_in, y_tgt, _ in train_loader:
            steps_in_epoch += 1

            warm = int(cfg.get('chroma_bias_warmup_epochs', 0))
            if hasattr(train_loader.dataset, 'set_bias_active'):
                train_loader.dataset.set_bias_active(epoch <= warm)

            x_in  = x_in.to(device, non_blocking=True, memory_format=torch.channels_last)    # [B,3,H,W] grayscale replicated
            y_tgt = y_tgt.to(device, non_blocking=True, memory_format=torch.channels_last)   # [B,3,H,W] true color

            # Chroma nudge to break grey copying
            if cfg.get("uv_input_dither", True) and net.training:
                with torch.no_grad():
                    y, u, v = rgb_to_yuv(x_in)  # (B,1,H,W)
                    std = float(cfg.get("uv_dither_std", 0.01))
                    if std > 0:
                        u = u + std * torch.randn_like(u)
                        v = v + std * torch.randn_like(v)
                        x_in = yuv_to_rgb(y, u, v, clamp=True).contiguous(memory_format=torch.channels_last)

            with autocast(enabled=use_amp):
                pred_rgb = net(x_in)
                loss_l1  = charbonnier(pred_rgb, y_tgt)

                pr_s = F.interpolate(pred_rgb,  size=lpips_side, mode="bilinear", align_corners=False)
                gt_s = F.interpolate(y_tgt,     size=lpips_side, mode="bilinear", align_corners=False)
                loss_lp = lpips_loss(pr_s, gt_s)
                
                loss_uv = chroma_weighted_uv_loss(pred_rgb, y_tgt, wmin=float(cfg.get('uv_wmin', 0.5)), wmax=float(cfg.get('uv_wmax', 2.0)))

                if lambda_uv_sched == 'cosine_hold':
                    lambda_uv_eff = cosine_hold_decay_lambda_uv(epoch, total_epochs, lambda_uv, lambda_uv_min, hold_pct=float(cfg.get('lambda_uv_hold_pct', 0.6)))
                elif lambda_uv_sched == 'cosine':
                    lambda_uv_eff = cosine_decay_lambda_uv(epoch, total_epochs, lambda_uv, lambda_uv_min)
                else:
                    lambda_uv_eff = dynamic_chroma_weighting(epoch, total_epochs, lambda_uv)

                loss_sat = saturation_prior(pred_rgb, y_tgt, tau=float(cfg.get('sat_tau', 0.05)))
                lam_sat = float(cfg.get('lambda_sat', 0.05))

                loss = (
                    w_l1 * loss_l1 +
                    w_lp * loss_lp +
                    lambda_uv_eff * loss_uv +
                    lam_sat * loss_sat
                ) / grad_accum


            scaler.scale(loss).backward()
            global_step += 1

            if global_step % grad_accum == 0:
                prev = opt._step_count  # PyTorch-internal counter
                scaler.step(opt)
                scaler.update()
                if opt._step_count > prev:  # only advance scheduler if we really stepped
                    sched.step()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(net.module if use_ddp else net)

            # lightweight log (rank 0 only)
            if is_main and global_step % log_every == 0:
                current_lr = opt.param_groups[0]["lr"]
                total_now = (w_l1 * loss_l1 + w_lp * loss_lp + lambda_uv_eff * loss_uv).item()
                tracker.log_train(global_step, loss_l1.item(), loss_lp.item(), loss_uv.item(), total_now, current_lr, lambda_uv_eff=float(lambda_uv_eff))
                print(f"[{epoch}] step={global_step} l1={loss_l1.item():.4f} lp={loss_lp.item():.4f} uv={loss_uv.item():.4f} lambda_uv={lambda_uv_eff:.3f} loss_total = {total_now:.4f} lr={current_lr:.2e}")

            # ------------- periodic sample panel (rank 0 only) -------------
            if is_main and (panel_every > 0) and (global_step % panel_every == 0):
                with torch.no_grad():
                    # Take the first sample in the current batch
                    x0  = x_in[0:1]                 # (1,3,H,W)
                    y0  = y_tgt[0:1].clamp(0, 1)    # GT
                    p0  = pred_rgb[0:1].clamp(0, 1) # prediction

                    # Build greyscale (Y) tile from input (robust even if input is replicated grey)
                    y_lum, _, _ = rgb_to_yuv(x0)    # (1,1,H,W)
                    gs3 = y_lum.repeat(1, 3, 1, 1).clamp(0, 1)

                    # Save panel
                    panel_dir = out_root / "panels"
                    panel_dir.mkdir(parents=True, exist_ok=True)
                    panel_path = panel_dir / f"step_{global_step:07d}.jpg"
                    save_panel_with_titles(
                        [gs3, y0, p0],
                        ["Greyscale", "Ground truth", "Model Prediction"],
                        panel_path
                    )

            # validate (rank 0 only)
            if is_main and val_every > 0 and global_step % val_every == 0:
                net.eval()
                if ema is not None:
                    bak = (net.module if use_ddp else net).state_dict()
                    ema.apply_to(net.module if use_ddp else net)

                # accumulators
                val_l1, val_lp, val_uv, val_total, n_count = 0.0, 0.0, 0.0, 0.0, 0

                with torch.no_grad():
                    for x_v, y_v, _ in eval_loader:
                        x_v  = x_v.to(device)
                        y_v  = y_v.to(device)

                        pred_v = net(x_v)

                        # L1/Charbonnier (same as train)
                        l1 = charbonnier(pred_v, y_v).item()

                        # LPIPS on resized tensors (same size used in train)
                        pr_s = F.interpolate(pred_v, size=lpips_side, mode="bilinear", align_corners=False)
                        gt_s = F.interpolate(y_v,   size=lpips_side, mode="bilinear", align_corners=False)
                        lp = lpips_loss(pr_s, gt_s).item()

                        # UV chroma-only term (same as train)
                        _, u1, v1 = rgb_to_yuv(pred_v)
                        _, u2, v2 = rgb_to_yuv(y_v)
                        uv = (F.l1_loss(u1, u2) + F.l1_loss(v1, v2)).item()

                        # per-epoch UV weight (match train-side choice, using total_epochs)
                        if lambda_uv_sched == "cosine":
                            lambda_uv_eff_v = cosine_decay_lambda_uv(epoch, total_epochs, lambda_uv, lambda_uv_min)
                        else:
                            lambda_uv_eff_v = dynamic_chroma_weighting(epoch, total_epochs, lambda_uv)

                        # accumulate raw components + weighted total
                        loss_sat_v = saturation_prior(pred_v, y_v, tau=float(cfg.get('sat_tau', 0.05)))
                        lam_sat = float(cfg.get('lambda_sat', 0.05))
                        val_l1   += l1
                        val_lp   += lp
                        val_uv   += uv
                        val_total += (w_l1 * l1) + (w_lp * lp) + (lambda_uv_eff_v * uv) + (lam_sat * loss_sat_v)

                # means
                avg_l1    = val_l1 / max(n_count, 1)
                avg_lp    = val_lp / max(n_count, 1)
                avg_uv    = val_uv / max(n_count, 1)
                avg_total = val_total / max(n_count, 1)

                if is_main:
                    tracker.log_val(global_step, avg_l1, avg_lp, avg_uv, avg_total)
                    print(f"[val] step={global_step} L1={avg_l1:.4f} LPIPS={avg_lp:.4f} UV={avg_uv:.4f} TOTAL={avg_total:.4f}")

                if ema is not None:
                    (net.module if use_ddp else net).load_state_dict(bak, strict=False)
                net.train()

                best_on = str(cfg.get('best_on', 'total')).lower()
                if (best_on == 'lpips' and avg_lp < getattr(main, 'best_lp', float('inf'))) or (best_on != 'lpips' and avg_total < best_total):
                    if best_on == 'lpips':
                        try:
                            main.best_lp = avg_lp
                        except Exception:
                            pass
                        if is_main:
                            save_ckpt(out_root/"best_lpips.ckpt", net.module if use_ddp else net,
                                    opt, scaler, global_step, best_total, ema)
                    else:
                        best_total = avg_total
                        if is_main:
                            save_ckpt(out_root/"best_total.ckpt", net.module if use_ddp else net,
                                    opt, scaler, global_step, best_total, ema)

            # periodic checkpoint (rank 0 only)
            if is_main and save_every > 0 and global_step % save_every == 0:
                save_ckpt(out_root/f"step_{global_step}.ckpt", net.module if use_ddp else net, opt, scaler, global_step, best_total, ema)

            if global_step % 25 == 0:
                torch.cuda.empty_cache()

        # ---- epoch timing end ----
        if is_main:
            epoch_dur = time.time() - epoch_start
            steps_per_sec = steps_in_epoch / max(epoch_dur, 1e-9)
            tracker.log_epoch(epoch=epoch, duration_sec=epoch_dur, steps=steps_in_epoch, steps_per_sec=steps_per_sec)
            print(f"[epoch {epoch}] duration={epoch_dur:.2f}s  steps={steps_in_epoch}  {steps_per_sec:.2f} steps/s")

    # ---- stage timing end ----
    if is_main:
        total_dur = time.time() - train_wall_start
        print(f"[stage done] wall_time={total_dur/60.0:.2f} min  ({total_dur:.2f}s)")

    return global_step, best_total

# --------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train_root", type=str, default=None)   # path to coco/train2017
    ap.add_argument("--val_root",   type=str, default=None)   # path to coco/val2017
    ap.add_argument("--ann_root",   type=str, default=None)   # path to coco/annotations
    ap.add_argument("--pretrained", type=str, default=None)
    ap.add_argument("--resume",     type=str, default=None)
    ap.add_argument("--exp_name",   type=str, default="exp")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    # allow CLI overrides
    for k in ("train_root","val_root","pretrained","resume","ann_root"):
        v = getattr(args, k)
        if v is not None: cfg[k] = v

    # DDP init
    world = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world > 1
    if use_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    set_seed(int(cfg.get("seed", 1337)))

    _lpips.to(device)

    exp_base = Path(cfg.get("out_dir", "outputs"))
    exp_name = args.exp_name or cfg.get("exp_defaults", "exp")
    ts = time.strftime("%Y%m%d-%H%M%S")  # e.g. 20251022-2038
    run_name = f"{exp_name}_{ts}"
    out_root: Path = exp_base / run_name


    is_main = (not use_ddp) or dist.get_rank() == 0
    if is_main:
        out_root.mkdir(parents=True, exist_ok=True)
        # save merged config for reproducibility
        yaml.safe_dump(cfg, open(out_root / "config_merged.yaml", "w"))
        # write a small run meta file
        with open(out_root / "run_meta.txt", "w") as f:
            cmd = " ".join(os.sys.argv)
            f.write(f"run_name: {run_name}\n")
            f.write(f"timestamp: {ts}\n")
            f.write(f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}\n")
            f.write(f"cmd: {cmd}\n")
        # best-effort symlink to latest run for this exp_name
        try:
            latest_link = exp_base / f"{exp_name}_latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(out_root.name)  # relative symlink
        except Exception:
            # fallback: write a text pointer if symlink not allowed
            with open(exp_base / f"{exp_name}_latest.txt", "w") as f:
                f.write(str(out_root.resolve()))

    # ------------------stats tracking----------------
    tracker = StatTracker(
        out_dir=out_root,
        redraw_every=int(cfg.get("plot_redraw_every", 50)),
        smoothing=float(cfg.get("plot_smoothing", 0.0)),
        is_main=is_main,
    )
    if is_main:
        tracker.start_run()  # mark run start and create epoch CSV

    # --------------------- data ---------------------
    train_loader, eval_loader, train_samp, eval_samp = build_coco_dataloaders(
        cfg,
        use_ddp=use_ddp,
        rank=(dist.get_rank() if use_ddp else 0),
    )

    # parent DS we will slice each epoch
    base_train_ds = train_loader.dataset

    # fixed pool built once
    train_pool_size = int(cfg.get("train_pool_size", 10000))
    train_pool_seed = int(cfg.get("train_pool_seed", 1337))
    pool_indices = sample_pool_indices(len(base_train_ds), train_pool_size, train_pool_seed)
    if is_main:
        print(f"Pool prepared: using {len(pool_indices)} images out of {len(base_train_ds)}")

    # --------------------- model ---------------------
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
    )
    if use_ddp:
        net = DDP(net, device_ids=[torch.cuda.current_device()], find_unused_parameters=False)

    torch.cuda.reset_peak_memory_stats() # More memory checking

    # ------------------- Scaler --------------------
    scaler = GradScaler(enabled=bool(cfg.get("amp", True)))

    # --------------------- EMA ---------------------
    ema = None
    if float(cfg.get("ema_decay", 0.0)) > 0:
        ema = EMA(net.module if use_ddp else net, decay=float(cfg["ema_decay"]))

    # --------------------- resume ---------------------
    step, best_total, ema_sd = 0, 1e9, None
    if cfg.get("resume"):
        if is_main:
            print(f"[resume] loading {cfg['resume']}")
        _model = net.module if use_ddp else net
        
        # Use a dummy optimizer for state restoration; real opt/sched are created per stage below
        opt_dummy = AdamW(_model.parameters(), lr=1e-6)
        step, best_total, ema_sd = load_ckpt(cfg["resume"], _model, opt_dummy, scaler)

        if ema_sd is not None and ema is not None:
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

    if stage1_epochs > 0:
        # ---- Stage 1: freeze all but heads + last K blocks ----
        print(f"\n=== Stage 1: freeze all but heads + last {last_k_blocks} blocks for {stage1_epochs} epochs ===")
        freeze_all_but_last(net.module if use_ddp else net, last_k_blocks=last_k_blocks)

        opt = AdamW(filter(lambda p: p.requires_grad, (net.parameters() if not use_ddp else net.module.parameters())),
                    lr=stage1_lr, weight_decay=weight_decay)
        # schedule steps from per-epoch subset size instead of full dataset length
        total_steps_s1 = (
            stage1_epochs
            * max(1, int(cfg.get("epoch_subset_size", 5000)))
            // max(1, int(cfg.get("batch_size", 10)))
        )
        sched = WarmupCosine(opt, base_lr=stage1_lr, warmup_steps=warmup_steps, max_steps=total_steps_s1, min_lr=min_lr)

        step, best_total = run_training_epochs(
            net=net, opt=opt, sched=sched, scaler=scaler, ema=ema,
            base_train_ds=base_train_ds, eval_loader=eval_loader,
            cfg=cfg, device=device, out_root=out_root, tracker=tracker,
            is_main=is_main, use_ddp=use_ddp, rank=rank,
            start_epoch=1, end_epoch=stage1_epochs, total_epochs=total_epochs,
            global_step=step, best_total=best_total, pool_indices=pool_indices
        )

    # ---- Stage 2: unfreeze all and continue ----
    if total_epochs > stage1_epochs:
        print(f"\n=== Stage 2: unfreeze ALL layers for remaining {total_epochs - stage1_epochs} epochs ===")
        for p in (net.module if use_ddp else net).parameters():
            p.requires_grad = True

        opt = AdamW((net.parameters() if not use_ddp else net.module.parameters()), lr=stage2_lr, weight_decay=weight_decay)
        
        total_steps_s2 = (
            (total_epochs - stage1_epochs)
            * max(1, int(cfg.get("epoch_subset_size", 5000)))
            // max(1, int(cfg.get("batch_size", 10)))
        )
        sched = WarmupCosine(opt, base_lr=stage2_lr, warmup_steps=warmup_steps, max_steps=total_steps_s2, min_lr=min_lr)

        step, best_total = run_training_epochs(
            net=net, opt=opt, sched=sched, scaler=scaler, ema=ema,
            base_train_ds=base_train_ds, eval_loader=eval_loader,
            cfg=cfg, device=device, out_root=out_root, tracker=tracker,
            is_main=is_main, use_ddp=use_ddp, rank=rank,
            start_epoch=stage1_epochs + 1, end_epoch=total_epochs, total_epochs=total_epochs,
            global_step=step, best_total=best_total, pool_indices=pool_indices
        )

    # ---- finish / teardown ----
    if is_main:
        tracker.save_fig()
        tracker.close()

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()