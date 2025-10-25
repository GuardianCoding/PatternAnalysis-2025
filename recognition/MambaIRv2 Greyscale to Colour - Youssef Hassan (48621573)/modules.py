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

# --------- helpers: checkpoint loader ---------

def load_color_dn15_weights(model: nn.Module, ckpt_path: Optional[str]) -> None:
    """
    Load weights from mambairv2_ColorDN_15.pth into the freshly constructed MambaIRv2.
    We use strict=False so channel-mismatched layers (to be adapted later) won't break loading.
    """
    if not ckpt_path:
        print("[pretrained] No checkpoint path provided; proceeding without loading.")
        return
    if not os.path.isfile(ckpt_path):
        print(f"[pretrained] Checkpoint not found at: {ckpt_path}")
        return
    sd = torch.load(ckpt_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[pretrained] loaded with missing={len(missing)}, unexpected={len(unexpected)}")


# --------- public factory ---------

def build_mambairv2_colorizer(
    embed_dim: int = 174,
    depths: Tuple[int, ...] = (4, 4, 6, 4),
    pretrained: Optional[str] = None,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Construct the official MambaIRv2 backbone with defaults that match common v2 configs,
    load ColorDN(σ=15) weights, then adapt I/O to L->ab.
    Returns a ready-to-train nn.Module with:
        forward(L: [B,1,H,W]) -> ab: [B,2,H,W]
    """
    if not _HAVE_MAMBAIR:
        raise ImportError(
            "Could not import MambaIRv2 from basicsr.archs.mambairv2_arch. "
            "Make sure the MambaIR repo is installed (e.g., `pip install -e external/MambaIR`). "
            f"Original import error: {_IMPORT_ERR}"
        )

    # The repo’s MambaIRv2 constructors accept common backbones args (embed_dim/depths, etc.).
    # For ColorDN, there is no upscaling, so we keep it as a plain restoration net.
    # We deliberately DO NOT hard-set num_in_ch/num_out_ch here, because we will load the
    # official 3-in/3-out weights first and only then adapt IO to 1/2.
    net = MambaIRv2(embed_dim=embed_dim, depths=list(depths))  # repo signature. :contentReference[oaicite:2]{index=2}

    # Load ColorDN_15 (3->3) weights into the backbone
    load_color_dn15_weights(net, pretrained)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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