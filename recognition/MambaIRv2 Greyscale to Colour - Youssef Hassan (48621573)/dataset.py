"""dataset.py — COCO2017 loaders and sampling utilities for grayscale→RGB colorization.

Overview
--------
This module provides:
  • Train/Eval PyTorch Datasets that read COCO2017 images and return
    (gray-3-channel input, RGB target) tensors in [0,1].
  • Optional COCO2017 *auto-download* (train/val images + annotations) with
    safe zip extraction (path traversal protected).
  • Deterministic *pool + per-epoch subset* sampling utilities that keep
    each epoch bounded while maintaining diversity over time.
  • DataLoader factory that is DDP-aware (DistributedSampler) and includes
    worker seeding for reproducible-ish augmentation across dataloader workers.

Why gray→RGB from RGB→RGB?
--------------------------
The training scheme uses standard RGB images as *targets* and feeds the model a
3-channel grayscale replicate as *input*. This lets you fine-tune an RGB→RGB
backbone for colorization without changing the architecture.

Key Ideas
---------
• Mild RGB jitter is applied to the *target only*, so the model can’t cheat by
  simply copying its input — it must learn a semantics-aware color mapping.
• Optional chroma-biased crops during warmup can help the model see more colorful
  regions early (controlled by `chroma_bias_try` + `chroma_bias_warmup_epochs`).
• Crop size is enforced on the shortest side so RandomCrop is always feasible.

DDP Notes
---------
Use `build_coco_dataloaders(cfg, use_ddp=True, rank=rank)` in distributed runs.
This will attach DistributedSampler to both train/eval and set per-epoch seeds.

Outputs
-------
Typical training sample tuple: (x_gray3, y_rgb)
  • x_gray3: torch.FloatTensor [3, H, W] in [0,1], luminance replicated to 3 channels
  • y_rgb  : torch.FloatTensor [3, H, W] in [0,1], optionally jittered

"""

from __future__ import annotations

import os
import random
import zipfile
from typing import Tuple, Optional, Dict, List

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset

from torchvision.datasets import CocoDetection
from torchvision.datasets.utils import download_url
from torchvision import transforms
import torchvision.transforms.functional as F

from PIL import Image
from pathlib import Path


__all__ = [
    "CocoColorisationTrain",
    "CocoColorisationEval",
    "build_coco_dataloaders",
    "sample_pool_indices",
    "sample_epoch_indices",
    "build_epoch_subset_loader",
]


# =============================================================================
# COCO auto-download configuration
# =============================================================================

COCO_2017_URLS: Dict[str, str] = {
    "train_imgs": "http://images.cocodataset.org/zips/train2017.zip",
    "val_imgs":   "http://images.cocodataset.org/zips/val2017.zip",
    "ann":        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def _exists(path: Optional[str]) -> bool:
    """Return True if `path` is a non-empty string and exists on disk."""
    return bool(path) and os.path.exists(path)


def _safe_extract(zip_path: str, dst_dir: str) -> None:
    """Safely extract a zip file ensuring all member paths stay within `dst_dir`.

    This prevents zip-slip attacks by rejecting members whose resolved path would
    escape the destination directory.

    Parameters
    ----------
    zip_path : str
        Path to the .zip archive to extract.
    dst_dir : str
        Destination directory (created if missing).
    """
    os.makedirs(dst_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = os.path.abspath(os.path.join(dst_dir, member.filename))
            root   = os.path.abspath(dst_dir)
            if not (target == root or target.startswith(root + os.sep)):
                raise RuntimeError(f"Unsafe path in zip: {member.filename}")
        zf.extractall(dst_dir)


def ensure_coco_2017(base_dir: str, need_train: bool = True, need_val: bool = True) -> Dict[str, str]:
    """Ensure COCO2017 assets exist; download and extract missing parts if needed.

    Returns a dict with absolute paths:
        { "train_root": ..., "val_root": ..., "ann_root": ... }

    Notes
    -----
    • Download happens only if corresponding folders/files are absent.
    • Annotations zip contains both train/val JSONs and must be extracted once.

    Parameters
    ----------
    base_dir : str
        Directory under which train2017/, val2017/, and annotations/ will live.
    need_train, need_val : bool
        Whether to ensure the respective image sets.

    Raises
    ------
    FileNotFoundError
        If annotations JSONs are not found after extraction.
    """
    os.makedirs(base_dir, exist_ok=True)
    train_root = os.path.join(base_dir, "train2017")
    val_root   = os.path.join(base_dir, "val2017")
    ann_root   = os.path.join(base_dir, "annotations")

    # 1) Annotations (contains both train/val JSONs)
    if not _exists(os.path.join(ann_root, "instances_train2017.json")) or not _exists(
        os.path.join(ann_root, "instances_val2017.json")
    ):
        zip_dst = os.path.join(base_dir, "annotations_trainval2017.zip")
        if not _exists(zip_dst):
            print("[COCO] Downloading annotations…")
            download_url(COCO_2017_URLS["ann"], base_dir, filename=os.path.basename(zip_dst))
        print("[COCO] Extracting annotations…")
        _safe_extract(zip_dst, base_dir)

    # 2) Train images
    if need_train and not _exists(train_root):
        zip_dst = os.path.join(base_dir, "train2017.zip")
        if not _exists(zip_dst):
            print("[COCO] Downloading train2017 images…")
            download_url(COCO_2017_URLS["train_imgs"], base_dir, filename=os.path.basename(zip_dst))
        print("[COCO] Extracting train2017…")
        _safe_extract(zip_dst, base_dir)

    # 3) Val images
    if need_val and not _exists(val_root):
        zip_dst = os.path.join(base_dir, "val2017.zip")
        if not _exists(zip_dst):
            print("[COCO] Downloading val2017 images…")
            download_url(COCO_2017_URLS["val_imgs"], base_dir, filename=os.path.basename(zip_dst))
        print("[COCO] Extracting val2017…")
        _safe_extract(zip_dst, base_dir)

    # Sanity check
    ann_train = os.path.join(ann_root, "instances_train2017.json")
    ann_val   = os.path.join(ann_root, "instances_val2017.json")
    if not _exists(ann_train) or not _exists(ann_val):
        raise FileNotFoundError("[COCO] Annotations not found after extraction.")

    return dict(train_root=train_root, val_root=val_root, ann_root=ann_root)


def _require_pycoco() -> None:
    """Ensure `pycocotools` is importable, raising a helpful message otherwise."""
    try:
        import pycocotools  # noqa: F401
    except Exception as e:
        raise ImportError(
            "pycocotools is required for CocoDetection.\n"
            "Install via: conda install -c conda-forge pycocotools   (or pip install pycocotools)"
        ) from e


# =============================================================================
# Image transforms / helpers
# =============================================================================

def _resize_min_side(img: Image.Image, min_side: int) -> Image.Image:
    """Resize PIL image so its shorter side is at least `min_side` (keeps aspect)."""
    w, h = img.size
    if min(w, h) >= min_side:
        return img
    scale = float(min_side) / min(w, h)
    return img.resize((int(w * scale + 0.5), int(h * scale + 0.5)), Image.BICUBIC)


def _random_longside_resize(img: Image.Image, target: int, scale_range: Tuple[float, float]) -> Image.Image:
    """Randomly scale image so its *long* side is near `target * s`, s~Uniform(scale_range)."""
    s = random.uniform(*scale_range)
    long_side = int(target * s)
    w, h = img.size
    if w >= h:
        new_w, new_h = long_side, int(h * long_side / w + 0.5)
    else:
        new_h, new_w = long_side, int(w * long_side / h + 0.5)
    return img.resize((new_w, new_h), Image.BICUBIC)


@torch.no_grad()
def _to_gray3_and_rgb(img_rgb: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return `(gray3_tensor, rgb_tensor)` from a PIL RGB image, both in [0,1].

    The grayscale tensor is produced via luminance conversion and replicated to 3 channels.
    Shapes: both tensors are [3, H, W].
    """
    t = F.to_tensor(img_rgb)                               # [3,H,W] in [0,1]
    g1 = F.rgb_to_grayscale(t, num_output_channels=1)      # [1,H,W]
    g3 = g1.repeat(3, 1, 1).contiguous()                   # [3,H,W]
    return g3, t.contiguous()


# =============================================================================
# Dataset classes
# =============================================================================

class CocoColorisationTrain(Dataset):
    """COCO train set: yields (gray3_input, rgb_target) tensors in [0,1].

    Augmentations
    -------------
    • Random long-side resize in range `(1.0, 1.15)` by default (keeps aspect).
    • Enforce min side ≥ crop_size, then RandomCrop ensures correct output size.
    • Optional horizontal flip.
    • Optional mild RGB jitter on the **target only** to prevent trivial copying.

    Chroma-biased crops (optional)
    ------------------------------
    Set `chroma_bias_try > 1` and call `set_bias_active(True)` to take multiple
    random crop candidates and pick the one with the highest mean per-pixel
    saturation proxy (max(channel) - min(channel)). Use during warmup only.

    Parameters
    ----------
    img_root : str
        Path to COCO train2017 images.
    ann_file : str
        Path to `annotations/instances_train2017.json`.
    crop_size : int
        Output crop size (square). Recommended divisible by 16.
    hflip : bool
        If True, apply random horizontal flip with p=0.5.
    rgb_jitter_prob : float
        Probability of applying mild RGB jitter to the target image.
    rgb_jitter_strength : float
        Jitter magnitude (brightness/contrast/saturation ∈ [1±s]).
    longside_scale_range : (float, float)
        Range for random long-side scaling prior to cropping.
    chroma_bias_try : int
        If >1, number of candidate crops to try when chroma bias is active.
    chroma_bias_warmup_epochs : int
        Unused here but stored for symmetry with config; activation is controlled
        externally by `set_bias_active()` during warmup.
    """

    def __init__(
        self,
        img_root: str,
        ann_file: str,
        crop_size: int = 256,
        hflip: bool = True,
        rgb_jitter_prob: float = 0.2,
        rgb_jitter_strength: float = 0.1,
        longside_scale_range: Tuple[float, float] = (1.00, 1.15),
        chroma_bias_try: int = 0,
        chroma_bias_warmup_epochs: int = 0,
    ):
        super().__init__()
        _require_pycoco()
        self.ds = CocoDetection(root=img_root, annFile=ann_file)

        self.crop_size = int(crop_size)
        self.hflip = bool(hflip)
        self.rgb_jitter_prob = float(rgb_jitter_prob)
        self._jitter_s = float(rgb_jitter_strength)
        self.scale_range = tuple(longside_scale_range)

        self._chroma_try = int(chroma_bias_try)
        self._chroma_warmup_epochs = int(chroma_bias_warmup_epochs)
        self._bias_active = True  # toggled by `set_bias_active()` from the training loop

    # ---- control hooks -------------------------------------------------------

    def set_bias_active(self, active: bool) -> None:
        """Enable/disable chroma-biased crop selection (called by trainer during warmup)."""
        self._bias_active = bool(active)

    # ---- standard dataset API ------------------------------------------------

    def __len__(self) -> int:
        return len(self.ds)

    def _jitter_rgb(self, img: Image.Image) -> Image.Image:
        """Apply mild brightness/contrast/saturation jitter to a PIL image."""
        s = self._jitter_s
        if s <= 0:
            return img
        b = 1.0 + random.uniform(-s, s)  # brightness
        img = F.adjust_brightness(img, b)
        c = 1.0 + random.uniform(-s, s)  # contrast
        img = F.adjust_contrast(img, c)
        t = F.to_tensor(img)             # saturation (tensor-based op)
        sat = 1.0 + random.uniform(-s, s)
        t = F.adjust_saturation(t, sat)
        return F.to_pil_image(t)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Produce a single training sample `(x_gray3, y_rgb)` in [0,1]."""
        img_pil, _ = self.ds[index]
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")

        # Random long-side scaling → ensure min side ≥ crop_size for RandomCrop.
        if self.scale_range is not None:
            try:
                img_pil = _random_longside_resize(img_pil, target=self.crop_size, scale_range=self.scale_range)
            except Exception:
                pass
        img_pil = _resize_min_side(img_pil, self.crop_size)

        # Chroma-biased crop selection (if active and >1 tries)
        best_crop, best_score = None, -1.0
        tries = self._chroma_try if (self._bias_active and self._chroma_try and self._chroma_try > 1) else 1
        for _ in range(tries):
            i, j, h, w = transforms.RandomCrop.get_params(img_pil, output_size=(self.crop_size, self.crop_size))
            cand = F.crop(img_pil, i, j, self.crop_size, self.crop_size)
            if tries == 1:
                best_crop = cand
                break
            t = F.to_tensor(cand)
            mx, mn = t.max(dim=0).values, t.min(dim=0).values
            sat = (mx - mn).mean().item()
            if sat > best_score:
                best_score, best_crop = sat, cand
        img_pil = best_crop

        # Optional flip
        if self.hflip and random.random() < 0.5:
            img_pil = F.hflip(img_pil)

        # Branch: unjittered input vs jittered target (to avoid identity mapping)
        img_input_pil  = img_pil
        img_target_pil = img_pil
        if random.random() < self.rgb_jitter_prob:
            img_target_pil = self._jitter_rgb(img_target_pil)

        x_in  = F.to_tensor(img_input_pil)          # [3,H,W], unjittered
        y_tgt = F.to_tensor(img_target_pil)         # [3,H,W], (maybe) jittered

        # Convert input to 3-channel grayscale (replicate luminance)
        x_gray = F.rgb_to_grayscale(x_in, num_output_channels=1).repeat(3, 1, 1)
        return x_gray, y_tgt


class CocoColorisationEval(Dataset):
    """COCO val set: yields `(gray3_input, rgb_target, filename)` for inspection/metrics.

    Center-crops each image to `crop_size` after ensuring min side ≥ crop_size.
    This keeps evaluation deterministic and comparable across runs.
    """

    def __init__(self, img_root: str, ann_file: str, crop_size: int = 256):
        super().__init__()
        _require_pycoco()
        self.ds = CocoDetection(root=img_root, annFile=ann_file)
        self.crop_size = int(crop_size)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, _ = self.ds[idx]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = _resize_min_side(img, self.crop_size)
        img = F.center_crop(img, [self.crop_size, self.crop_size])

        x_in, y_tgt = _to_gray3_and_rgb(img)
        img_id = self.ds.ids[idx]
        file_name = self.ds.coco.loadImgs(img_id)[0]["file_name"]
        return x_in, y_tgt, file_name


# =============================================================================
# DataLoader factory (DDP-aware)
# =============================================================================

def _worker_init_fn(worker_id: int) -> None:
    """Seed Python's RNG inside each worker so augmentations are deterministically varied."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)


def build_coco_dataloaders(cfg: dict, use_ddp: bool = False, rank: int = 0):
    """Build train/eval DataLoaders with optional COCO auto-download and deterministic subsets.

    Config Keys Used
    ----------------
    auto_download : bool
        If True, download/extract COCO into `coco_root` when local roots are missing.
    coco_root : str
        Parent folder of `train2017/`, `val2017/`, `annotations/` (used when auto_download).
    train_root, val_root, ann_root : str
        Explicit paths (skip auto-download). If any are missing and `auto_download` is not set,
        `coco_root` will be used as a fallback.
    crop_size, hflip, rgb_jitter_prob, rgb_jitter_strength, longside_scale_range
        Augmentation knobs for `CocoColorisationTrain`.
    train_max_items / val_max_items : int
        If >0, take a fixed subset of each split (deterministic by seed).
    train_subset_seed / val_subset_seed : int
        Seeds used to form the fixed subset order.
    num_workers, num_workers_val, batch_size, val_batch_size, prefetch_factor
        DataLoader knobs.

    Returns
    -------
    (train_loader, eval_loader, train_sampler, eval_sampler)
        Samplers are `None` when not using DDP.
    """
    auto_dl = bool(cfg.get("auto_download", False))
    coco_root = cfg.get("coco_root", None)

    roots_missing = not (cfg.get("train_root") and cfg.get("val_root") and cfg.get("ann_root"))
    if auto_dl or roots_missing:
        if coco_root is None:
            coco_root = os.path.abspath("./datasets/coco")
        need_train = cfg.get("need_train", True)
        need_val   = cfg.get("need_val", True)
        ensured = ensure_coco_2017(coco_root, need_train=need_train, need_val=need_val)
        train_root = ensured["train_root"]; val_root = ensured["val_root"]; ann_root = ensured["ann_root"]
    else:
        train_root = cfg.get("train_root", "./datasets/coco/train2017")
        val_root   = cfg.get("val_root",   "./datasets/coco/val2017")
        ann_root   = cfg.get("ann_root",   "./datasets/coco/annotations")

    crop_size  = int(cfg.get("crop_size", 256))
    assert crop_size % 16 == 0, f"crop_size={crop_size} should be divisible by 16 for stable down/upsampling"

    train_ds = CocoColorisationTrain(
        img_root=train_root,
        ann_file=os.path.join(ann_root, "instances_train2017.json"),
        crop_size=crop_size,
        hflip=bool(cfg.get("hflip", True)),
        rgb_jitter_prob=float(cfg.get("rgb_jitter_prob", 0.2)),
        rgb_jitter_strength=float(cfg.get("rgb_jitter_strength", 0.1)),
        longside_scale_range=tuple(cfg.get("longside_scale_range", (1.00, 1.15))),
        chroma_bias_try=int(cfg.get("chroma_bias_try", 0)),
        chroma_bias_warmup_epochs=int(cfg.get("chroma_bias_warmup_epochs", 0)),
    )

    # Optional fixed-size training subset (good for quick iterations)
    train_max_items = int(cfg.get("train_max_items", 0))
    if train_max_items > 0 and train_max_items < len(train_ds):
        seed = int(cfg.get("train_subset_seed", 1337))
        rng = random.Random(seed)
        idxs = list(range(len(train_ds)))
        rng.shuffle(idxs)
        idxs = sorted(idxs[:train_max_items])
        train_ds = Subset(train_ds, idxs)

    eval_ds = CocoColorisationEval(
        img_root=val_root,
        ann_file=os.path.join(ann_root, "instances_val2017.json"),
        crop_size=crop_size,
    )

    val_max_items = int(cfg.get("val_max_items", 0))
    if val_max_items > 0 and val_max_items < len(eval_ds):
        seed = int(cfg.get("val_subset_seed", 1337))
        rng = random.Random(seed)
        idxs = list(range(len(eval_ds)))
        rng.shuffle(idxs)
        idxs = sorted(idxs[:val_max_items])
        eval_ds = Subset(eval_ds, idxs)

    train_sampler: Optional[DistributedSampler] = None
    eval_sampler: Optional[DistributedSampler]  = None
    if use_ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
        eval_sampler  = DistributedSampler(eval_ds,  shuffle=False, drop_last=False)

    nworkers_train = max(1, int(cfg.get("num_workers", 6)))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 10)),
        shuffle=(not use_ddp),
        sampler=train_sampler,
        num_workers=nworkers_train,
        pin_memory=True,
        persistent_workers=(nworkers_train > 0),
        prefetch_factor=int(cfg.get("prefetch_factor", 4)),
        worker_init_fn=_worker_init_fn,
    )

    nworkers_val = max(1, int(cfg.get("num_workers_val", 4)))
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(cfg.get("val_batch_size", 8)),
        shuffle=False,
        sampler=eval_sampler,
        num_workers=nworkers_val,
        pin_memory=True,
        persistent_workers=(nworkers_val > 0),
        worker_init_fn=_worker_init_fn,
    )

    return train_loader, eval_loader, train_sampler, eval_sampler


# =============================================================================
# Pool + per-epoch subset helpers
# =============================================================================

def sample_pool_indices(ds_len: int, pool_size: int, seed: int) -> List[int]:
    """Pick a fixed *pool* of indices (deterministic, sorted) for the whole run."""
    pool_size = max(0, min(int(pool_size), int(ds_len)))
    rng = random.Random(int(seed))
    idxs = list(range(ds_len))
    rng.shuffle(idxs)
    return sorted(idxs[:pool_size])


def sample_epoch_indices(pool_indices: List[int], subset_size: int, seed: int, epoch: int) -> List[int]:
    """Pick a different *subset* (deterministic, sorted) per epoch from the fixed pool."""
    if not pool_indices:
        return []
    subset_size = max(0, min(int(subset_size), len(pool_indices)))
    rng = random.Random(int(seed) + int(epoch))
    idxs = list(pool_indices)
    rng.shuffle(idxs)
    return sorted(idxs[:subset_size])


def build_epoch_subset_loader(
    base_train_ds: Dataset,
    epoch_indices: List[int],
    cfg: dict,
    use_ddp: bool,
    rank: int,
):
    """Build a DataLoader over `Subset(base_train_ds, epoch_indices)`.

    This is used by the trainer to rebuild a small but fresh loader each epoch.

    Returns
    -------
    loader : DataLoader
        Epoch subset loader with shuffling/DistributedSampler configured.
    train_sampler : Optional[DistributedSampler]
        Non-None only when `use_ddp=True`.
    """
    subset = Subset(base_train_ds, epoch_indices)
    train_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(
            subset,
            shuffle=True,
            drop_last=False,
            rank=rank,
            num_replicas=dist.get_world_size() if torch.distributed.is_initialized() else 1,
        )

    nworkers_train = max(1, int(cfg.get("num_workers", 6)))
    loader = DataLoader(
        subset,
        batch_size=int(cfg.get("batch_size", 10)),
        shuffle=(not use_ddp),
        sampler=train_sampler,
        num_workers=nworkers_train,
        pin_memory=True,
        persistent_workers=(nworkers_train > 0),
        prefetch_factor=int(cfg.get("prefetch_factor", 4)),
        worker_init_fn=_worker_init_fn,
    )
    return loader, train_sampler