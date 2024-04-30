import random
import numpy as np
import torch
from torch import nn


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_int(x):
    if isinstance(x, torch.Tensor):
        return int(x.item())
    return int(x)


def replace_module_fn(model: nn.Module, mod_class, fn):
    for mod in model.modules():
        for name, child in mod.named_children():
            if isinstance(child, mod_class):
                new_child = fn(child)
                new_child.original = (child,)
                mod.add_module(name, new_child)
                pass


def freeze_model(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def pad_to_square(x: torch.Tensor, value=None):
    h, w = x.shape[-2:]
    if h > w:
        pad = (h - w) // 2 + (h - w) % 2
        return torch.nn.functional.pad(x, (pad, pad, 0, 0,), value=value)[..., :h, :h]
    elif w > h:
        pad = (w - h) // 2 + (w - h) % 2
        return torch.nn.functional.pad(x, (0, 0, pad, pad), value=value)[..., :w, :w]
    else:
        return x


class PadToSquare:
    def __init__(self, value=None):
        self.value = value

    def __call__(self, x: torch.Tensor):
        return pad_to_square(x, self.value)

