import torch
from torch import nn
from torchvision.transforms import Compose

from ultralytics.utils.checks import check_requirements


def get_yolow_clip(device=None):
    try:
        import clip
    except ImportError:
        check_requirements("git+https://github.com/openai/CLIP.git")
        import clip

    model, tf = clip.load("ViT-B/32", jit=False, device=device)
    # device = next(model.parameters()).device
    # text_token = clip.tokenize(text).to(device)
    # txt_feats = model.encode_text(text_token).to(dtype=torch.float32)
    # txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
    # self.txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1]).detach()
    # self.model[-1].nc = len(text)
    return model, tf


def get_clip_encoders(use_normalize_transform, device=None):
    model, tf = get_yolow_clip()
    import clip

    class ClipImgEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

            # Resize(n_px, interpolation=BICUBIC),
            # CenterCrop(n_px),
            # _convert_image_to_rgb,
            # ToTensor(),
            # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),

            if use_normalize_transform:
                self.tf = Compose([*tf.transforms[:2], tf.transforms[-1]])
            else:    
                # discard the normalize transform as dataset images are already normalized
                self.tf = Compose(tf.transforms[:2])

        def __call__(self, x):
            return self.model.encode_image(self.tf(x))
    
    class ClipTxtEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def __call__(self, x, device=None):
            txt_tensor = clip.tokenize(x).to(device=device)
            return self.model.encode_text(txt_tensor)

    return ClipImgEncoder().to(device), ClipTxtEncoder().to(device)