import torch
from torch.nn import Module

def load_mambair_ckpt(model: Module, path: str):
    if not path: 
        print("[Pretrained] None provided."); return
    sd = torch.load(path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[Pretrained] loaded with missing={len(missing)}, unexpected={len(unexpected)}")