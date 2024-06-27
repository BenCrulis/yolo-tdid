import os
from pathlib import Path
import argparse

from PIL import Image

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms.functional import to_tensor, to_pil_image

from torch.jit import script

from models.yolow.utils import get_clip_encoders, get_yolow_clip

from tdid.aggregations import average_embeddings


default_transform = Compose([ToTensor(), Resize((336, 336))])


class AllImagesDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.paths = sorted(root.glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path)
        if self.transform:
            img = self.transform(img)
        return img


def parse_args():
    parser = argparse.ArgumentParser(description='Compute embeddings')
    parser.add_argument('dataset', type=Path, help='dataset root')
    parser.add_argument('--bs', default=5, type=int, help="batch size")
    parser.add_argument('--dry', action='store_true', help="dry run (don't save anything)")
    parser.add_argument('--device', type=str, default="cuda:0", help='choose device (default to cuda:0 if available)')
    return parser.parse_args()


def compute_embeddings(root, batch_size=5, dry=False, device=None):
    # model, txt_enc = get_clip_encoders(True, device=device)
    model, tf = get_yolow_clip()
    model.to(device)
    model.eval()
    
    ds = AllImagesDataset(root, transform=tf)
    eval_iterator = DataLoader(ds, batch_size=batch_size, shuffle=False)

    using_tqdm = False
    try:
        import tqdm
        eval_iterator = tqdm.tqdm(eval_iterator)
        using_tqdm = True
    except ImportError:
        print("tqdm not available, not using progress bar")

    embeddings = []

    with torch.inference_mode():
        for x in eval_iterator:
            x = x.to(device)
            
            emb = model.encode_image(x)
            embeddings.append(emb)
            pass
    
    if not dry:
        embeddings = torch.cat(embeddings, dim=0)
        torch.save(embeddings, "embeddings.pt")
        print("Embeddings saved")
    pass


def main():
    args = parse_args()
    print(args)
    bs = args.bs
    dry = args.dry
    if dry: print("WARNING, running in dry mode, embeddings will not be saved.")
    root = args.dataset

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device", device)

    compute_embeddings(
        root,
        batch_size=bs,
        dry=dry,
        device=device)
    print("end")


if __name__ == "__main__":
    main()
