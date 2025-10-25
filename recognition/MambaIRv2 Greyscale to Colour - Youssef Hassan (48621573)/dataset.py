# dataset.py
import os, random, zipfile
from typing import Tuple, Optional, Dict
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.datasets.utils import download_url
from PIL import Image

__all__ = [
    "CocoColorisationTrain",
    "CocoColorisationEval",
    "build_coco_dataloaders",
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
    """
    PIL RGB -> (gray3, rgb) tensors
        gray3: [3,H,W] in [0,1] (grayscale replicated to 3 channels)
        rgb:   [3,H,W] in [0,1] (ground-truth color)
    """
    # target: ground-truth color
    rgb = F.to_tensor(img_rgb).contiguous()          # [3,H,W], float32, [0,1]
    # input: grayscale replicated to 3 channels (model expects 3-ch input)
    gray3_pil = F.rgb_to_grayscale(img_rgb, num_output_channels=3)
    gray3 = F.to_tensor(gray3_pil).contiguous()      # [3,H,W], float32, [0,1]
    return gray3, rgb

# ------------------------ datasets ------------------------

class CocoColorisationTrain(Dataset):
    def __init__(self,
                 img_root: str,
                 ann_file: str,
                 crop_size: int = 256,
                 hflip: bool = True,
                 rgb_jitter_prob: float = 0.2,
                 rgb_jitter_strength: float = 0.1,
                 longside_scale_range: Tuple[float, float] = (1.00, 1.15)):
        super().__init__()
        _require_pycoco()
        self.ds = CocoDetection(root=img_root, annFile=ann_file)
        self.crop_size = int(crop_size)
        self.hflip = bool(hflip)
        self.rgb_jitter_prob = float(rgb_jitter_prob)
        self._jitter_s = float(rgb_jitter_strength)
        self.scale_range = tuple(longside_scale_range)

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

    def __getitem__(self, idx: int):
        img, _ = self.ds[idx]
        if img.mode != "RGB":
            img = img.convert("RGB")

        img = _random_longside_resize(img, self.crop_size, self.scale_range)
        img = _resize_min_side(img, self.crop_size)

        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.crop_size, self.crop_size))
        img = F.crop(img, i, j, self.crop_size, self.crop_size)

        if self.hflip and random.random() < 0.5:
            img = F.hflip(img)

        if random.random() < self.rgb_jitter_prob:
            img = self._jitter_rgb(img)

        # convert to (gray3 input, rgb target) tensors
        x_in, y_tgt = _to_gray3_and_rgb(img)         # [3,H,W], [3,H,W]

        img_id = self.ds.ids[idx]

        file_name = self.ds.coco.loadImgs(img_id)[0]["file_name"]
        
        return x_in, y_tgt, file_name


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
    )
    eval_ds = CocoColorisationEval(
        img_root=val_root,
        ann_file=os.path.join(ann_root, "instances_val2017.json"),
        crop_size=crop_size,
    )

    train_sampler: Optional[DistributedSampler] = None
    eval_sampler: Optional[DistributedSampler]  = None
    if use_ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
        eval_sampler  = DistributedSampler(eval_ds,  shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 10)),
        shuffle=(not use_ddp),
        sampler=train_sampler,
        num_workers=int(cfg.get("num_workers", 6)),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=int(cfg.get("prefetch_factor", 4)),
        worker_init_fn=_worker_init_fn,
    )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(cfg.get("val_batch_size", 8)),
        shuffle=False,
        sampler=eval_sampler,
        num_workers=max(1, int(cfg.get("num_workers_val", 4))),
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=_worker_init_fn,
    )

    return train_loader, eval_loader, train_sampler, eval_sampler