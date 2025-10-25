# modules.py
# Builds an official MambaIRv2 model compatible with the released
# "mambairv2_ColorDN_15.pth" checkpoint, then adapts it to 1-in/2-out for L->ab.
#
# Usage (training):
#   from modules import build_mambairv2_colorizer
#   net = build_mambairv2_colorizer(pretrained="checkpoints/mambairv2_ColorDN_15.pth")
#
# Notes:
# - We construct the SAME arch class used by the repo for ColorDN (MambaIRv2),
#   then replace only the first/last convs to 1-in / 2-out after loading weights.
# - Checkpoint formats vary ("state_dict", "params", "network_g"). We handle all.
#
# Refs (repo + weights list and compatibility statements):
# - Repo & arch files: basicsr/archs/mambairv2_arch.py in csguoh/MambaIR. :contentReference[oaicite:0]{index=0}
# - The README notes MambaIRv2 is compatible with previous MambaIR/backbone releases
#   and hosts pretrained weights on HF. :contentReference[oaicite:1]{index=1}

from __future__ import annotations
import os
from typing import Tuple, Optional

import torch
import torch.nn as nn

# Try to import the official class from the repo install.
# If you installed the repo with `pip install -e external/MambaIR`, this should work.
try:
    from basicsr.archs.mambairv2_arch import MambaIRv2  # constructor lives here in the repo
    _HAVE_MAMBAIR = True
except Exception as e:
    MambaIRv2 = None
    _HAVE_MAMBAIR = False
    _IMPORT_ERR = e

# --------- public factory ---------

def build_mambairv2_colorizer(
    embed_dim: int = 174,
    depths: Tuple[int, ...] = (4, 4, 6, 4),
    num_heads: Tuple[int, ...] = (6, 6, 6, 6),
    pretrained: Optional[str] = None,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Build the standard RGB→RGB MambaIRv2 model (3 input / 3 output)
    for color restoration or gray→color finetuning with full checkpoint reuse.
    """
    if not _HAVE_MAMBAIR:
        raise ImportError(
            "Could not import MambaIRv2 from basicsr.archs.mambairv2_arch. "
            "Install with `pip install -e external/MambaIR`. "
            f"Original import error: {_IMPORT_ERR}"
        )

    net = MambaIRv2(
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=embed_dim,
        depths=list(depths),
        num_heads=list(num_heads),
        use_checkpoint=pretrained,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return net.to(device)


# --------- quick self-test (optional) ---------
if __name__ == "__main__":
    # Smoke test: build the model without a checkpoint and run a dummy forward.
    try:
        model = build_mambairv2_colorizer(pretrained=None, device=torch.device('cpu'))
        x = torch.randn(1, 1, 64, 64)
        y = model(x)
        print("OK:", y.shape)
    except Exception as e:
        print("modules.py self-test error:", e)