"""modules.py — Model factory for MambaIRv2 colorization (RGB→RGB finetuning).

Overview
--------
This module exposes a single factory, `build_mambairv2_colorizer`, which:
  • Instantiates the official MambaIRv2 restoration backbone from `basicsr`.
  • Leaves the network architecture unchanged (RGB→RGB), allowing us to train
    gray→RGB by simply feeding 3-channel grayscale inputs during training.
  • Optionally loads a *compatible* checkpoint with `strict=False` to tolerate
    key mismatches between repos or variants.

Why keep RGB→RGB?
-----------------
Keeping the canonical interface avoids risky surgery in attention/state-space
blocks. The trainer handles gray replication and color losses externally.

Checkpoints
-----------
The loader accepts common container formats (e.g., {"state_dict": ...},
{"network_g": ...}). Missing/unexpected keys are printed for transparency.

Memory Layout
-------------
The model is moved to `channels_last` (NHWC) for better Tensor Core utilization
on Ampere+ GPUs — this usually yields a small speedup for convolutions.

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
    MambaIRv2 = None  # type: ignore[assignment]
    _HAVE_MAMBAIR = False
    _IMPORT_ERR = e


# =============================================================================
# Public factory
# =============================================================================

def build_mambairv2_colorizer(
    upscale: int = 1,
    in_chans: int = 3,
    img_size: int = 128,
    img_range: float = 1.0,
    embed_dim: int = 174,
    d_state: int = 16,
    depths: Tuple[int, ...] = (4, 4, 6, 4),
    num_heads: Tuple[int, ...] = (6, 6, 6, 6),
    window_size: int = 16,
    inner_rank: int = 64,
    num_tokens: int = 128,
    convffn_kernel_size: int = 5,
    mlp_ratio: float = 2.0,
    pretrained: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Build standard RGB→RGB MambaIRv2 suitable for gray→color finetuning.

    Parameters
    ----------
    upscale : int
        Kept at 1 for colorization (no SR). Provided for completeness.
    in_chans : int
        Number of input channels; must be 3 for the official MambaIRv2 RGB path.
    img_size : int
        Nominal training crop (not strictly enforced at runtime).
    img_range : float
        Expected dynamic range (1.0 for [0,1] tensors).
    embed_dim, d_state, depths, num_heads, window_size, inner_rank, num_tokens,
    convffn_kernel_size, mlp_ratio : assorted hyperparameters
        Passed straight through to `MambaIRv2`.
    pretrained : str | None
        Optional path to a checkpoint to load with `strict=False` for tolerance.
    device : torch.device | None
        If None, cuda is used when available.

    Returns
    -------
    nn.Module
        The model on the requested device in channels_last memory format.

    Raises
    ------
    ImportError
        If `MambaIRv2` cannot be imported from `basicsr.archs.mambairv2_arch`.
    """
    if not _HAVE_MAMBAIR:
        raise ImportError(
            "Could not import MambaIRv2 from basicsr.archs.mambairv2_arch.\n"
            "Install with `pip install -e external/MambaIR` (or the package that provides it).\n"
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
        use_checkpoint=True,  # gradient checkpointing (reduce activation memory)
    )

    # Optional checkpoint: handle common container keys and load loosely
    if pretrained:
        sd = torch.load(pretrained, map_location="cpu")
        if isinstance(sd, dict):
            for k in ("state_dict", "params", "network_g", "model"):
                if k in sd and isinstance(sd[k], dict):
                    sd = sd[k]
                    break
        missing, unexpected = net.load_state_dict(sd, strict=False)
        print(f"[pretrained] strict=False  missing={len(missing)}  unexpected={len(unexpected)}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return net.to(device, memory_format=torch.channels_last)


# =============================================================================
# Quick self-test (optional)
# =============================================================================

if __name__ == "__main__":
    try:
        model = build_mambairv2_colorizer(pretrained=None, device=torch.device("cpu"))
        x = torch.randn(1, 3, 64, 64)
        y = model(x)
        print("OK:", y.shape)
    except Exception as e:
        print("modules.py self-test error:", e)