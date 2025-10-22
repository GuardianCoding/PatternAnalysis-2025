# dataset.py
import os, random
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from PIL import Image
import kornia

__all__ = [
    "CocoColorisationTrain",
    "CocoColorisationEval",
    "build_coco_dataloaders",
]

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

def _to_L_ab(img_rgb: Image.Image):
    rgb = F.to_tensor(img_rgb).unsqueeze(0)          # (1,3,H,W)
    lab = kornia.color.rgb_to_lab(rgb)               # L [0,100], ab ~ [-128,127]
    L   = lab[:, :1] / 100.0
    ab  = lab[:, 1:] / 128.0
    return L.squeeze(0), ab.squeeze(0)

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
        self.ds = CocoDetection(img_root=img_root, annFile=ann_file)
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

        i, j, h, w = F.get_params(img, output_size=(self.crop_size, self.crop_size))
        img = F.crop(img, i, j, self.crop_size, self.crop_size)

        if self.hflip and random.random() < 0.5:
            img = F.hflip(img)

        if random.random() < self.rgb_jitter_prob:
            img = self._jitter_rgb(img)

        L, ab = _to_L_ab(img)

        img_id = self.ds.ids[idx]
        file_name = self.ds.coco.loadImgs(img_id)[0]["file_name"]
        return L, ab, file_name


class CocoColorisationEval(Dataset):
    def __init__(self,
                 img_root: str,
                 ann_file: str,
                 crop_size: int = 256):
        super().__init__()
        self.ds = CocoDetection(img_root=img_root, annFile=ann_file)
        self.crop_size = int(crop_size)

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx: int):
        img, _ = self.ds[idx]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = _resize_min_side(img, self.crop_size)
        img = F.center_crop(img, [self.crop_size, self.crop_size])

        L, ab = _to_L_ab(img)

        img_id = self.ds.ids[idx]
        file_name = self.ds.coco.loadImgs(img_id)[0]["file_name"]
        return L, ab, file_name

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

    Expected cfg keys (with defaults):
      - train_root: "./datasets/coco/train2017"
      - val_root:   "./datasets/coco/val2017"
      - ann_root:   "./datasets/coco/annotations"
      - crop_size:  256
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
    train_root = cfg.get("train_root", "./datasets/coco/train2017")
    val_root   = cfg.get("val_root",   "./datasets/coco/val2017")
    ann_root   = cfg.get("ann_root",   "./datasets/coco/annotations")
    crop_size  = int(cfg.get("crop_size", 256))

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