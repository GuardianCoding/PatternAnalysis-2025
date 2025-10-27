import warnings # Ignore warnings from lpip module
warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*deprecated.*")

import os, argparse, time
from pathlib import Path
import yaml
import torch
import torch.distributed as dist
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
import torch.nn.functional as F

from dataset import build_coco_dataloaders
from modules import build_mambairv2_colorizer
from utils.metrics import set_seed, lpips_loss
from utils.train_tracker import StatTracker
from utils.checkpoint_io import save_ckpt, load_ckpt

# Memory savings
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

# --------------------- utilities ---------------------
class EMA:
    def __init__(self, model: Module, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for p in self.shadow.values():
            p.requires_grad = False
    @torch.no_grad()
    def update(self, model):
        if self.decay <= 0: return
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
    @torch.no_grad()
    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict=False)

class WarmupCosine:
    def __init__(self, optimizer, base_lr, warmup_steps, max_steps):
        self.opt = optimizer
        self.base = float(base_lr)
        self.warm = max(1, int(warmup_steps))
        self.max_steps = max_steps
        self.t = 0
    def step(self):
        self.t += 1
        if not self.max_steps or self.max_steps <= self.warm:
            lr = self.base
        elif self.t <= self.warm:
            lr = self.base * self.t / self.warm
        else:
            progress = (self.t - self.warm) / (self.max_steps - self.warm)
            lr = 0.5 * self.base * (1 + torch.cos(torch.tensor(progress * 3.1415926535)).item())
        for g in self.opt.param_groups:
            g["lr"] = lr

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

    # --------------------- losses/optim/sched ---------------------
    def charbonnier(x, y, eps=1e-3):
        return torch.mean(torch.sqrt((x - y)**2 + eps**2))

    opt = AdamW(net.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    scaler = GradScaler(enabled=bool(cfg.get("amp", True)))

    total_steps = None
    if cfg.get("epochs") and len(train_loader) > 0:
        total_steps = int(cfg["epochs"]) * len(train_loader) // max(1, int(cfg.get("grad_accum", 1)))
    sched = WarmupCosine(opt, base_lr=float(cfg["lr"]), warmup_steps=int(cfg.get("warmup_steps", 1000)), max_steps=total_steps)

    # --------------------- EMA ---------------------
    ema = None
    if float(cfg.get("ema_decay", 0.0)) > 0:
        ema = EMA(net.module if use_ddp else net, decay=float(cfg["ema_decay"]))

    # --------------------- resume ---------------------
    step, best_lp, ema_sd = 0, 1e9, None
    if cfg.get("resume"):
        if is_main:
            print(f"[resume] loading {cfg['resume']}")
        _model = net.module if use_ddp else net
        step, best_lp, ema_sd = load_ckpt(cfg["resume"], _model, opt, scaler)
        if ema_sd is not None and ema is not None:
            ema.shadow = ema_sd
        if is_main:
            print(f"[resume] step={step} best_lpips={best_lp:.4f}")

    # --------------------- train ---------------------
    grad_accum = max(1, int(cfg.get("grad_accum", 1)))
    save_every = int(cfg.get("save_every", 2000))
    val_every  = int(cfg.get("val_every", 1000))
    epochs     = int(cfg.get("epochs", 10))
    lpips_side = int(cfg.get("lpips_side", min(int(cfg.get("crop_size", 128)), 192)))

    if use_ddp and train_samp is not None:
        train_samp.set_epoch(1)

    train_wall_start = time.time() # total run start (wall)

    for epoch in range(1, epochs + 1):
        if use_ddp and train_samp is not None:
            train_samp.set_epoch(epoch)

        # ---- epoch timing start ----
        epoch_start = time.time()
        steps_in_epoch = 0

        net.train()
        opt.zero_grad(set_to_none=True)

        for x_in, y_tgt, _ in train_loader:
            steps_in_epoch += 1      # per-epoch step counter

            x_in  = x_in.to(device, non_blocking=True, memory_format=torch.channels_last)    # [B,3,H,W] grayscale replicated
            y_tgt = y_tgt.to(device, non_blocking=True, memory_format=torch.channels_last)   # [B,3,H,W] true color

            with autocast(enabled=bool(cfg.get("amp", True))):
                pred_rgb = net(x_in)
                loss_l1  = charbonnier(pred_rgb, y_tgt)
                pr_s = F.interpolate(pred_rgb,  size=lpips_side, mode="bilinear", align_corners=False)
                gt_s = F.interpolate(y_tgt,     size=lpips_side, mode="bilinear", align_corners=False)
                loss_lp = lpips_loss(pr_s, gt_s)
                loss = (loss_l1 + loss_lp) / grad_accum

            scaler.scale(loss).backward()
            step += 1

            if step % grad_accum == 0:
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                if ema is not None:
                    ema.update(net.module if use_ddp else net)

            # lightweight log (rank 0 only)
            if is_main and step % int(cfg.get("log_every", 100)) == 0:
                current_lr = opt.param_groups[0]["lr"]
                tracker.log_train(step, loss_l1.item(), loss_lp.item(), (loss_l1 + loss_lp).item(), current_lr)
                print(f"[{epoch}] step={step} l1={loss_l1.item():.4f} lp={loss_lp.item():.4f} lr={opt.param_groups[0]['lr']:.2e}")

            # validate (rank 0 only)
            if is_main and step % val_every == 0:
                net.eval()
                if ema is not None:
                    bak = (net.module if use_ddp else net).state_dict()
                    ema.apply_to(net.module if use_ddp else net)

                val_l1, val_lp, n_count = 0.0, 0.0, 0
                with torch.no_grad():
                    for x_in, y_tgt, _ in eval_loader:
                        x_in  = x_in.to(device)
                        y_tgt = y_tgt.to(device)
                        pred_rgb = net(x_in)
                        val_l1  += charbonnier(pred_rgb, y_tgt).item()
                        val_lp  += lpips_loss(pred_rgb, y_tgt).item()
                        n_count += 1

                avg_l1 = val_l1 / max(n_count, 1)
                avg_lp = val_lp / max(n_count, 1)
                if is_main:
                    tracker.log_val(step, avg_lp)
                print(f"[val] step={step} CHARBONNIER={avg_l1:.4f} LPIPS={avg_lp:.4f}")

                if ema is not None:
                    (net.module if use_ddp else net).load_state_dict(bak, strict=False)
                net.train()

                if avg_lp < best_lp:
                    best_lp = avg_lp
                    save_ckpt(out_root/"best_lpips.ckpt", net.module if use_ddp else net, opt, scaler, step, best_lp, ema)

            # periodic checkpoint (rank 0 only)
            if is_main and step % save_every == 0:
                save_ckpt(out_root/f"step_{step}.ckpt", net.module if use_ddp else net, opt, scaler, step, best_lp, ema)

            if step % 25 == 0:
                torch.cuda.empty_cache()

        # ---- epoch timing end ----
        if is_main:
            epoch_dur = time.time() - epoch_start
            steps_per_sec = steps_in_epoch / max(epoch_dur, 1e-9)
            tracker.log_epoch(epoch=epoch, duration_sec=epoch_dur,
                              steps=steps_in_epoch, steps_per_sec=steps_per_sec)
            print(f"[epoch {epoch}] duration={epoch_dur:.2f}s  steps={steps_in_epoch}  {steps_per_sec:.2f} steps/s")

    # ---- total timing end ----
    if is_main:
        total_dur = time.time() - train_wall_start
        tracker.end_run(total_duration_sec=total_dur)
        print(f"[training done] total_wall_time={total_dur/60.0:.2f} min  ({total_dur:.2f}s)")
        tracker.save_fig()
        tracker.close()

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()