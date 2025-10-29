import torch
import torch.nn.functional as F
from math import log10
import lpips

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