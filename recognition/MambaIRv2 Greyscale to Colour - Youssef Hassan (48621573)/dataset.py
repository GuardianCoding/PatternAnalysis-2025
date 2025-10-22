import random
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from PIL import Image
import kornia


def _resize_min_side(img: Image.Image, min_side: int) -> Image.Image:
    """Resize so the shortest side is >= min_side, preserving aspect ratio."""
    w, h = img.size
    if min(w, h) >= min_side:
        return img
    scale = float(min_side) / min(w, h)
    return img.resize((int(w * scale + 0.5), int(h * scale + 0.5)), Image.BICUBIC)


def _random_longside_resize(img: Image.Image, target: int, scale_range: Tuple[float, float]) -> Image.Image:
    """Resize so the *long* side is target * s, s∈[a,b], preserving aspect ratio."""
    s = random.uniform(*scale_range)
    long_side = int(target * s)
    w, h = img.size
    if w >= h:
        new_w, new_h = long_side, int(h * long_side / w + 0.5)
    else:
        new_h, new_w = long_side, int(w * long_side / h + 0.5)
    return img.resize((new_w, new_h), Image.BICUBIC)


def _to_L_ab(img_rgb: Image.Image):
    """Convert PIL RGB -> normalized (L, ab) tensors using kornia (L in [0,1], ab ~ [-1,1])."""
    rgb = F.to_tensor(img_rgb).unsqueeze(0)               # (1,3,H,W)
    lab = kornia.color.rgb_to_lab(rgb)                    # L in [0,100], ab ~ [-128,127]
    L   = lab[:, :1] / 100.0
    ab  = lab[:, 1:] / 128.0
    return L.squeeze(0), ab.squeeze(0)


class CocoColorisationTrain(Dataset):
    """
    COCO colorisation dataset (training).
    - Uses CocoDetection, ignores labels.
    - Geometric augs applied identically to gray and RGB by operating on PIL first:
        * random long-side resize (scale_range)
        * ensure min side >= crop_size
        * random crop (crop_size x crop_size)
        * random horizontal flip
        * optional mild RGB jitter (brightness/contrast/saturation)
    - Returns: (L, ab, filename)
    """
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
        self.crop_size = crop_size
        self.hflip = hflip
        self.rgb_jitter_prob = rgb_jitter_prob
        self.rgb_jitter = torch.nn.Sequential(  # lightweight, reproducible via F if preferred
            # Use torchvision ColorJitter via functional for deterministic? Keeping simple here:
        )
        # Store strengths for F.adjust_* use
        self._jitter_s = rgb_jitter_strength
        self.scale_range = longside_scale_range

    def __len__(self):
        return len(self.ds)

    def _jitter_rgb(self, img: Image.Image) -> Image.Image:
        # Apply simple jitter via functional to avoid global RNG changes.
        s = self._jitter_s
        # Brightness/contrast/saturation ±s
        if s <= 0:
            return img
        if random.random() < 1.0:  # apply all three in a random order
            # brightness
            b = 1.0 + random.uniform(-s, s)
            img = F.adjust_brightness(img, b)
            # contrast
            c = 1.0 + random.uniform(-s, s)
            img = F.adjust_contrast(img, c)
            # saturation (convert to tensor for F.adjust_saturation which expects tensor)
            tensor = F.to_tensor(img)
            sat = 1.0 + random.uniform(-s, s)
            tensor = F.adjust_saturation(tensor, sat)
            img = F.to_pil_image(tensor)
        return img

    def __getitem__(self, idx: int):
        img, _ = self.ds[idx]
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Geometric augs
        img = _random_longside_resize(img, self.crop_size, self.scale_range)
        img = _resize_min_side(img, self.crop_size)

        i, j, h, w = F.get_params(img, output_size=(self.crop_size, self.crop_size))
        img = F.crop(img, i, j, self.crop_size, self.crop_size)

        if self.hflip and random.random() < 0.5:
            img = F.hflip(img)

        # Optional color jitter (RGB only; target remains RGB)
        if random.random() < self.rgb_jitter_prob:
            img = self._jitter_rgb(img)

        # Convert to L/ab
        L, ab = _to_L_ab(img)

        # Filename (for tracking)
        img_id = self.ds.ids[idx]
        file_name = self.ds.coco.loadImgs(img_id)[0]["file_name"]

        return L, ab, file_name


class CocoColorisationEval(Dataset):
    """
    COCO colorisation dataset (evaluation/validation).
    - Deterministic: resize shortest side >= crop_size, center crop, no flip, no jitter.
    - Returns: (L, ab, filename)
    """
    def __init__(self,
                 img_root: str,
                 ann_file: str,
                 crop_size: int = 256):
        super().__init__()
        self.ds = CocoDetection(img_root=img_root, annFile=ann_file)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.ds)

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
