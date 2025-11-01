# dataset.py
import os, random, zipfile
from typing import Tuple, Optional, Dict
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.datasets.utils import download_url
import torch.distributed as dist
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

# ------------------------ URLs & auto-download helpers ------------------------

COCO_2017_URLS: Dict[str, str] = {
    "train_imgs": "http://images.cocodataset.org/zips/train2017.zip",
    "val_imgs":   "http://images.cocodataset.org/zips/val2017.zip",
    "ann":        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

def _exists(path: str) -> bool:
    return path is not None and os.path.exists(path)

def _safe_extract(zip_path: str, dst_dir: str):
    # Simple safe extract: ensure target stays under dst_dir
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.infolist():
            target = os.path.abspath(os.path.join(dst_dir, member.filename))
            if not target.startswith(os.path.abspath(dst_dir) + os.sep) and target != os.path.abspath(dst_dir):
                raise RuntimeError(f"Unsafe path in zip: {member.filename}")
        zf.extractall(dst_dir)

def ensure_coco_2017(base_dir: str, need_train: bool = True, need_val: bool = True) -> Dict[str, str]:
    """
    Ensure COCO 2017 train/val images and annotations exist under base_dir.
    Returns dict with keys: train_root, val_root, ann_root.
    """
    os.makedirs(base_dir, exist_ok=True)
    train_root = os.path.join(base_dir, "train2017")
    val_root   = os.path.join(base_dir, "val2017")
    ann_root   = os.path.join(base_dir, "annotations")

    # Download annotations if missing
    if not _exists(os.path.join(ann_root, "instances_train2017.json")) or not _exists(os.path.join(ann_root, "instances_val2017.json")):
        zip_dst = os.path.join(base_dir, "annotations_trainval2017.zip")
        if not _exists(zip_dst):
            print("[COCO] Downloading annotations...")
            download_url(COCO_2017_URLS["ann"], base_dir, filename=os.path.basename(zip_dst))
        print("[COCO] Extracting annotations...")
        os.makedirs(ann_root, exist_ok=True)
        _safe_extract(zip_dst, base_dir)

    # Download train images if missing
    if need_train and not _exists(train_root):
        zip_dst = os.path.join(base_dir, "train2017.zip")
        if not _exists(zip_dst):
            print("[COCO] Downloading train2017 images...")
            download_url(COCO_2017_URLS["train_imgs"], base_dir, filename=os.path.basename(zip_dst))
        print("[COCO] Extracting train2017...")
        _safe_extract(zip_dst, base_dir)

    # Download val images if missing
    if need_val and not _exists(val_root):
        zip_dst = os.path.join(base_dir, "val2017.zip")
        if not _exists(zip_dst):
            print("[COCO] Downloading val2017 images...")
            download_url(COCO_2017_URLS["val_imgs"], base_dir, filename=os.path.basename(zip_dst))
        print("[COCO] Extracting val2017...")
        _safe_extract(zip_dst, base_dir)

    # Final sanity
    ann_train = os.path.join(ann_root, "instances_train2017.json")
    ann_val   = os.path.join(ann_root, "instances_val2017.json")
    if not _exists(ann_train) or not _exists(ann_val):
        raise FileNotFoundError("[COCO] Annotations not found after extraction.")

    return dict(train_root=train_root, val_root=val_root, ann_root=ann_root)

def _require_pycoco():
    try:
        import pycocotools  # noqa: F401
    except Exception as e:
        raise ImportError(
            "pycocotools is required for CocoDetection. "
            "Install via: conda install -c conda-forge pycocotools   (or pip install pycocotools)"
        ) from e

# ------------------------ util funcs ------------------------

def _resize_min_side(img: Image.Image, min_side: int) -> Image.Image:
    w, h = img.size
    if min(w, h) >= min_side:
        return img
    scale = float(min_side) / min(w, h)
    return img.resize((int(w * scale + 0.5), int(h * scale + 0.5)), Image.BICUBIC)

def _random_longside_resize(img: Image.Image, target: int, scale_range: Tuple[float, float]) -> Image.Image:
    s = random.uniform(*scale_range)
    long_side = int(target * s)
    w, h = img.size
    if w >= h:
        new_w, new_h = long_side, int(h * long_side / w + 0.5)
    else:
        new_h, new_w = long_side, int(w * long_side / h + 0.5)
    return img.resize((new_w, new_h), Image.BICUBIC)

# --- RGB→RGB colorization I/O helper -----------------------------------------
@torch.no_grad()
def _to_gray3_and_rgb(img_rgb: Image.Image):
    # tensor path only (functional.rgb_to_grayscale expects a Tensor)
    t = F.to_tensor(img_rgb)                     # [3,H,W] in [0,1]
    g1 = F.rgb_to_grayscale(t, num_output_channels=1)  # [1,H,W]
    g3 = g1.repeat(3, 1, 1).contiguous()        # [3,H,W]
    return g3, t.contiguous()

# ------------------------ datasets ------------------------

class CocoColorisationTrain(Dataset):
    def __init__(self,
                img_root: str,
                ann_file: str,
                crop_size: int = 256,
                hflip: bool = True,
                rgb_jitter_prob: float = 0.2,
                rgb_jitter_strength: float = 0.1,
                longside_scale_range: Tuple[float, float] = (1.00, 1.15),
                chroma_bias_try: int = 0,            # how many random crops to try; pick the most colorful
                chroma_bias_warmup_epochs: int = 0,  # enable bias only for first N epochs
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
        self._bias_active = True

    def set_bias_active(self, active: bool):
        self._bias_active = bool(active)

    def __len__(self): return len(self.ds)

    def _jitter_rgb(self, img: Image.Image) -> Image.Image:
        s = self._jitter_s
        if s <= 0: return img
        # brightness
        b = 1.0 + random.uniform(-s, s)
        img = F.adjust_brightness(img, b)
        # contrast
        c = 1.0 + random.uniform(-s, s)
        img = F.adjust_contrast(img, c)
        # saturation (tensor path)
        t = F.to_tensor(img)
        sat = 1.0 + random.uniform(-s, s)
        t = F.adjust_saturation(t, sat)
        return F.to_pil_image(t)

    def __getitem__(self, index: int):
        img_pil, _ = self.ds[index]
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")

        # --- optional random longside scaling (data scale jitter) ---
        if self.scale_range is not None:
            try:
                img_pil = _random_longside_resize(img_pil, target=self.crop_size, scale_range=self.scale_range)
            except Exception:
                pass  # stay robust if cfg omitted it

        # --- guarantee crop feasibility: min side >= crop_size ---
        img_pil = _resize_min_side(img_pil, self.crop_size)

        # --- chroma-biased crop selection (only if enabled) ---
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
            sat = (mx - mn).mean().item()  # simple saturation proxy
            if sat > best_score:
                best_score, best_crop = sat, cand
        img_pil = best_crop

        # --- shared spatial aug ---
        if self.hflip and random.random() < 0.5:
            img_pil = F.hflip(img_pil)

        # -------- branch: unjittered input vs jittered target --------
        img_input_pil  = img_pil
        img_target_pil = img_pil
        if random.random() < self.rgb_jitter_prob:
            img_target_pil = self._jitter_rgb(img_target_pil)
        # --------------------------------------------------------------

        # --- to tensors ---
        x_in  = F.to_tensor(img_input_pil)         # [3,H,W] (unjittered)
        y_tgt = F.to_tensor(img_target_pil)        # [3,H,W] (jittered)

        # grayscale input (replicate luminance)
        x_gray = F.rgb_to_grayscale(x_in, num_output_channels=1).repeat(3, 1, 1)

        return x_gray, y_tgt

class CocoColorisationEval(Dataset):
    def __init__(self,
                 img_root: str,
                 ann_file: str,
                 crop_size: int = 256):
        super().__init__()
        _require_pycoco()
        self.ds = CocoDetection(root=img_root, annFile=ann_file)
        self.crop_size = int(crop_size)

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx: int):
        img, _ = self.ds[idx]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = _resize_min_side(img, self.crop_size)
        img = F.center_crop(img, [self.crop_size, self.crop_size])

        # convert to (gray3 input, rgb target) tensors
        x_in, y_tgt = _to_gray3_and_rgb(img)

        img_id = self.ds.ids[idx]
        
        file_name = self.ds.coco.loadImgs(img_id)[0]["file_name"]
        
        return x_in, y_tgt, file_name

# ------------------------ dataloader factory ------------------------

def _worker_init_fn(worker_id: int):
    # make random ops reproducible-ish across workers
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)

def build_coco_dataloaders(
    cfg: dict,
    use_ddp: bool = False,
    rank: int = 0,
):
    """
    Build train/eval dataloaders and (optional) samplers from a config dict.

    You can either supply explicit roots:
      - train_root, val_root, ann_root

    Or set:
      - coco_root: "/where/to/keep/coco"
      - auto_download: true

    Other keys (with defaults):
      - crop_size: 256
      - hflip: True
      - rgb_jitter_prob: 0.2
      - rgb_jitter_strength: 0.1
      - longside_scale_range: (1.0, 1.15)
      - batch_size: 10
      - val_batch_size: 8
      - num_workers: 6
      - num_workers_val: 4
      - prefetch_factor: 4
    """
    # Decide whether to auto-download
    auto_dl = bool(cfg.get("auto_download", False))
    coco_root = cfg.get("coco_root", None)

    # If explicit roots are missing or auto_download requested, ensure assets exist.
    roots_missing = not (cfg.get("train_root") and cfg.get("val_root") and cfg.get("ann_root"))
    if auto_dl or roots_missing:
        if coco_root is None:
            coco_root = os.path.abspath("./datasets/coco")  # default location
        need_train = cfg.get("need_train", True)
        need_val   = cfg.get("need_val", True)
        ensured = ensure_coco_2017(coco_root, need_train=need_train, need_val=need_val)
        train_root = ensured["train_root"]
        val_root   = ensured["val_root"]
        ann_root   = ensured["ann_root"]
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
        chroma_bias_warmup_epochs=int(cfg.get("chroma_bias_warmup_epochs", 0))
    )

    # ---- optional: cap training set size with a deterministic subset ----
    train_max_items = int(cfg.get("train_max_items", 0))
    if train_max_items > 0 and train_max_items < len(train_ds):
        seed = int(cfg.get("train_subset_seed", 1337))
        rng = random.Random(seed)
        idxs = list(range(len(train_ds)))
        rng.shuffle(idxs)
        idxs = sorted(idxs[:train_max_items])  # stable order for nicer logs
        train_ds = Subset(train_ds, idxs)

    eval_ds = CocoColorisationEval(
        img_root=val_root,
        ann_file=os.path.join(ann_root, "instances_val2017.json"),
        crop_size=crop_size,
    )

    # ---- optional: cap validation set size with a deterministic subset ----
    val_max_items = int(cfg.get("val_max_items", 0))
    if val_max_items > 0 and val_max_items < len(eval_ds):
        seed = int(cfg.get("val_subset_seed", 1337))
        rng = random.Random(seed)
        idxs = list(range(len(eval_ds)))
        rng.shuffle(idxs)
        idxs = sorted(idxs[:val_max_items])  # keep stable, increasing order for nice logs
        eval_ds = Subset(eval_ds, idxs)

    train_sampler: Optional[DistributedSampler] = None
    eval_sampler: Optional[DistributedSampler]  = None
    if use_ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
        eval_sampler  = DistributedSampler(eval_ds,  shuffle=False, drop_last=False)

    nworkers_train = max(1, int(cfg.get("num_workers", 4)))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 10)),
        shuffle=(not use_ddp),
        sampler=train_sampler,
        num_workers=int(cfg.get("num_workers", 6)),
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
        num_workers=max(1, int(cfg.get("num_workers_val", 4))),
        pin_memory=True,
        persistent_workers=(nworkers_val > 0),
        worker_init_fn=_worker_init_fn,
    )

    return train_loader, eval_loader, train_sampler, eval_sampler

# ------------------------ epoch-wise subset helpers ------------------------
def sample_pool_indices(ds_len: int, pool_size: int, seed: int) -> list[int]:
    """
    Deterministically pick a fixed pool of indices from a dataset of length ds_len.
    Returns a sorted list of length <= pool_size.
    """
    pool_size = max(0, min(int(pool_size), int(ds_len)))
    rng = random.Random(int(seed))
    idxs = list(range(ds_len))
    rng.shuffle(idxs)
    return sorted(idxs[:pool_size])

def sample_epoch_indices(pool_indices: list[int], subset_size: int, seed: int, epoch: int) -> list[int]:
    """
    Deterministically pick a different random subset from the fixed pool for each epoch.
    Uses (seed + epoch) so all ranks agree, then returns a sorted list for stable logs.
    """
    if not pool_indices:
        return []
    subset_size = max(0, min(int(subset_size), len(pool_indices)))
    rng = random.Random(int(seed) + int(epoch))
    idxs = list(pool_indices)
    rng.shuffle(idxs)
    return sorted(idxs[:subset_size])

def build_epoch_subset_loader(base_train_ds, epoch_indices: list[int], cfg: dict, use_ddp: bool, rank: int,):
    """
    Build a DataLoader over an epoch-specific Subset(base_train_ds, epoch_indices).
    DDP-safe via DistributedSampler.
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