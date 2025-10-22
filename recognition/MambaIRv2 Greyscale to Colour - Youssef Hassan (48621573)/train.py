import os, argparse, time
from pathlib import Path
import yaml
import torch
import torch.distributed as dist
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW

from dataset import build_coco_dataloaders
from modules import build_mambairv2_colorizer
from utils.metrics import set_seed, lab_to_rgb, lpips_loss
from utils.train_tracker import StatTracker
from utils.checkpoint_io import save_ckpt, load_ckpt

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

    out_root: Path = Path(cfg.get("out_dir", "outputs")) / (args.exp_name or cfg.get("exp_defaults","exp"))
    is_main = (not use_ddp) or dist.get_rank() == 0
    if is_main:
        out_root.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(cfg, open(out_root/"config_merged.yaml", "w"))

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
        embed_dim=cfg["model"]["embed_dim"],
        depths=tuple(cfg["model"]["depths"]),
        pretrained=cfg.get("pretrained"),
        device=device
    )
    if use_ddp:
        net = DDP(net, device_ids=[torch.cuda.current_device()], find_unused_parameters=False)

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

        for L, ab, _ in train_loader:
            steps_in_epoch += 1      # per-epoch step counter

            L, ab = L.to(device, non_blocking=True), ab.to(device, non_blocking=True)

            with autocast(enabled=bool(cfg.get("amp", True))):
                pred_ab = net(L)
                loss_l1 = charbonnier(pred_ab, ab) * float(cfg.get("lambda_l1", 1.0))
                pred_rgb = lab_to_rgb(L, pred_ab)
                tgt_rgb  = lab_to_rgb(L, ab)
                loss_lp  = lpips_loss(pred_rgb, tgt_rgb) * float(cfg.get("lambda_lpips", 1.0))
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

                lp_sum, n_cnt = 0.0, 0
                with torch.no_grad():
                    for Lv, abv, _names in eval_loader:
                        Lv, abv = Lv.to(device, non_blocking=True), abv.to(device, non_blocking=True)
                        pab = net(Lv)
                        pr = lab_to_rgb(Lv, pab)
                        gt = lab_to_rgb(Lv, abv)
                        lp = lpips_loss(pr, gt).item()
                        lp_sum += lp; n_cnt += 1
                avg_lp = lp_sum / max(n_cnt, 1)
                if is_main:
                    tracker.log_val(step, avg_lp)
                print(f"[val] step={step} LPIPS={avg_lp:.4f}")

                if ema is not None:
                    (net.module if use_ddp else net).load_state_dict(bak, strict=False)
                net.train()

                if avg_lp < best_lp:
                    best_lp = avg_lp
                    save_ckpt(out_root/"best_lpips.ckpt", net.module if use_ddp else net, opt, scaler, step, best_lp, ema)

            # periodic checkpoint (rank 0 only)
            if is_main and step % save_every == 0:
                save_ckpt(out_root/f"step_{step}.ckpt", net.module if use_ddp else net, opt, scaler, step, best_lp, ema)

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