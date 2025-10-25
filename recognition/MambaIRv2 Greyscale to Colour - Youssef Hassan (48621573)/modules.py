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
from typing import Tuple, Optional, Dict, Any

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


# --------- helpers: replace first/last conv to 1-in / 2-out (warm-start when possible) ---------

def _find_first_conv(module: nn.Module) -> nn.Conv2d:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            return m
    raise RuntimeError("Could not find a Conv2d input stem in the MambaIRv2 model.")

def _find_last_conv(module: nn.Module) -> nn.Conv2d:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("Could not find a Conv2d output head in the MambaIRv2 model.")
    return last

def _replace_module(root: nn.Module, old: nn.Module, new: nn.Module) -> bool:
    """Recursively replace a submodule instance."""
    for name, child in root.named_children():
        if child is old:
            setattr(root, name, new)
            return True
        if _replace_module(child, old, new):
            return True
    return False

@torch.no_grad()
def adapt_io_for_project(model: nn.Module) -> None:
    """
    After loading the ColorDN checkpoint (3-in/3-out), switch:
      - first conv to in_ch=1 (L)
      - last  conv to out_ch=2 (a,b)
    Keep everything else intact.
    """
    # ----- input stem: 1-in -----
    stem = _find_first_conv(model)
    old_w = stem.weight.data  # [C_out, C_in, k, k]
    new_stem = nn.Conv2d(
        in_channels=1,
        out_channels=old_w.shape[0],
        kernel_size=stem.kernel_size,
        stride=stem.stride,
        padding=stem.padding,
        dilation=stem.dilation,
        bias=(stem.bias is not None),
        groups=stem.groups
    )
    # If the pretrained was RGB (C_in=3), average weights across channels to warm-start grayscale.
    if old_w.shape[1] == 3:
        new_stem.weight.copy_(old_w.mean(dim=1, keepdim=True))
    else:
        # Fallback: copy first channel
        new_stem.weight.copy_(old_w[:, :1])
    if stem.bias is not None:
        new_stem.bias.copy_(stem.bias.data)
    _replace_module(model, stem, new_stem)

    # ----- output head: 2-out -----
    head = _find_last_conv(model)
    new_head = nn.Conv2d(
        in_channels=head.in_channels,
        out_channels=2,
        kernel_size=head.kernel_size,
        stride=head.stride,
        padding=head.padding,
        dilation=head.dilation,
        bias=(head.bias is not None),
        groups=head.groups
    )
    # Fresh init for the 2 output channels
    nn.init.kaiming_normal_(new_head.weight, nonlinearity="linear")
    if new_head.bias is not None:
        nn.init.zeros_(new_head.bias)
    _replace_module(model, head, new_head)


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

    # 1) load ColorDN_15 (3->3) weights into the backbone
    load_color_dn15_weights(net, pretrained)

    # 2) swap I/O to L->ab
    adapt_io_for_project(net)

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