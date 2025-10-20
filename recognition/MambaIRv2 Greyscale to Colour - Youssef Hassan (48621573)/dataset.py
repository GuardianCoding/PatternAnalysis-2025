import os, random
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import kornia

class ColorisationDataset(Dataset):
    def __init__(self, root, crop_size=256, hflip=True,
                 rgb_jitter_prob=0.2, rgb_jitter_strength=0.1):
        self.files = sorted([p for ext in ("*.jpg","*.png","*.jpeg","*.bmp")
                             for p in glob(os.path.join(root, "**", ext), recursive=True)])
        self.crop_size = crop_size
        self.hflip = hflip
        self.rgb_jitter_prob = rgb_jitter_prob
        self.rgb_jitter = T.ColorJitter(
            brightness=rgb_jitter_strength,
            contrast=rgb_jitter_strength,
            saturation=rgb_jitter_strength,
            hue=0.0
        )
        self.to_tensor = T.ToTensor()

    def __len__(self): return len(self.files)

    def _random_crop(self, img):
        w, h = img.size
        if w < self.crop_size or h < self.crop_size:
            scale = self.crop_size / min(w, h)
            img = img.resize((int(w*scale+0.5), int(h*scale+0.5)), Image.BICUBIC)
            w, h = img.size
        x0 = random.randint(0, w - self.crop_size)
        y0 = random.randint(0, h - self.crop_size)
        return img.crop((x0, y0, x0 + self.crop_size, y0 + self.crop_size))

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        img = self._random_crop(img)

        if self.hflip and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < self.rgb_jitter_prob:
            img = self.rgb_jitter(img)

        rgb = self.to_tensor(img).unsqueeze(0)     # (1,3,H,W) for kornia
        lab = kornia.color.rgb_to_lab(rgb)         # L in [0,100], ab ~ [-128,127]
        L   = lab[:, :1] / 100.0                   # -> [0,1]
        ab  = lab[:, 1:] / 128.0                   # ~[-1,1]

        return L.squeeze(0), ab.squeeze(0), os.path.basename(path)
