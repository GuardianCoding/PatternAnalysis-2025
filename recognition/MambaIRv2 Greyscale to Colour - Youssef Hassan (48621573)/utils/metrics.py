import torch
import torch.nn.functional as F
import kornia
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

def lpips_loss(a, b):
    with torch.no_grad():
        return _lpips(a*2-1, b*2-1).mean()