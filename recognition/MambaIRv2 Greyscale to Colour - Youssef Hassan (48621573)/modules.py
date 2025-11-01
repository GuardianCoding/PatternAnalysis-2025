"""modules.py — Model factory for MambaIRv2 colorization (RGB→RGB).
Builds the official MambaIRv2 arch and optionally loads a compatible checkpoint.
"""

from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn as nn

# Try to import the official class from the repo install.
try:
    from basicsr.archs.mambairv2_arch import MambaIRv2  # constructor in the repo
    _HAVE_MAMBAIR = True
except Exception as e:
    MambaIRv2 = None
    _HAVE_MAMBAIR = False
    _IMPORT_ERR = e

# --------- public factory ---------

def build_mambairv2_colorizer(
    upscale: int = 1,
    in_chans: int = 3,
    img_size: int = 128,
    img_range: float = 1.,
    embed_dim: int = 174,
    d_state: int = 16,
    depths: Tuple[int, ...] = (4, 4, 6, 4),
    num_heads: Tuple[int, ...] = (6, 6, 6, 6),
    window_size: int = 16,
    inner_rank: int = 64,
    num_tokens: int = 128,
    convffn_kernel_size: int = 5,
    mlp_ratio: float = 2.,
    pretrained: Optional[str] = None,
    device: Optional[torch.device] = None
) -> nn.Module:
    """Build standard RGB→RGB MambaIRv2 (finetune for gray→color without surgery)."""
    if not _HAVE_MAMBAIR:
        raise ImportError(
            "Could not import MambaIRv2 from basicsr.archs.mambairv2_arch. "
            "Install with `pip install -e external/MambaIR`. "
            f"Original import error: {_IMPORT_ERR}"
        )

    net = MambaIRv2(
        upscale=upscale,
        in_chans=in_chans,
        img_size=img_size,
        img_range=img_range,
        d_state=d_state,
        window_size=window_size,
        inner_rank=inner_rank,
        num_tokens=num_tokens,
        convffn_kernel_size=convffn_kernel_size,
        mlp_ratio=mlp_ratio,
        embed_dim=embed_dim,
        depths=list(depths),
        num_heads=list(num_heads),
        use_checkpoint=True,
    )

    # Optional checkpoint: handle common container keys and load loosely
    if pretrained:
        sd = torch.load(pretrained, map_location="cpu")
        if isinstance(sd, dict):
            for k in ("state_dict", "params", "network_g", "model"):
                if k in sd and isinstance(sd[k], dict):
                    sd = sd[k]; break
        missing, unexpected = net.load_state_dict(sd, strict=False)
        print(f"[pretrained] strict=False  missing={len(missing)}  unexpected={len(unexpected)}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return net.to(device, memory_format=torch.channels_last)


# --------- quick self-test (optional) ---------
if __name__ == "__main__":
    try:
        model = build_mambairv2_colorizer(pretrained=None, device=torch.device('cpu'))
        x = torch.randn(1, 3, 64, 64)
        y = model(x)
        print("OK:", y.shape)
    except Exception as e:
        print("modules.py self-test error:", e)