import torch
from torch.nn import Module
from pathlib import Path

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