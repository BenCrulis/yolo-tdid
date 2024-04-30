from collections import defaultdict

import torch
from torch.nn.functional import normalize
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.transforms import Compose, ToTensor, Resize

from utils.pytorch import pad_to_square, PadToSquare
from utils.boxes import crop_from_normalized_bb
from utils.nms import non_max_suppression2

from .commons import TDID
from .prompts import DetectObjectPrompt


default_transform = Compose([ToTensor(), PadToSquare(), Resize(21*32)])


class TDIDFromCrops(TDID):
    def __init__(self, model, prompt: DetectObjectPrompt, img_encoder, embedding_aggregator, margin=5, img_transform=default_transform, augmentations=False):
        self.embeddings = defaultdict(list)
        self.model = model
        self.prompt = prompt
        self.img_encoder = img_encoder
        self.embedding_aggregator = embedding_aggregator
        self.margin = margin
        self.img_transform = img_transform
        self.augmentations = augmentations

        self.debias_normalization = False

        # save training data outputs
        self.train_outputs = None
        self.train_targets = None
        self.bias = 0.0

    def _add_to_storage(self, nms, im, y):
        detect_prob = nms[0, 4]
        cropped = crop_from_normalized_bb(pad_to_square(to_tensor(im), value=0.5), nms[0, :4], margin=self.margin, square=True)
        cropped = pad_to_square(cropped, value=0.5)
        cropped = cropped.to(self.model.device)

        batch = [cropped]
        if self.augmentations:
            # add augmented versions of the cropped image with rotations and flips
            batch.append(torch.flip(cropped, [2]))
            rotated90 = torch.rot90(cropped, 1, [1, 2])
            batch.append(rotated90)
            rotated90anti = torch.rot90(cropped, -1, [1, 2])
            batch.append(rotated90anti)
        
        batch = torch.stack(batch, dim=0)

        emb = self.img_encoder(batch)
        l = self.embeddings[y]
        for e in emb:
            l.append(e)
        
    def _detect_and_crop(self, img, y):
        with torch.no_grad():
            if self.img_transform is not None:
                im_tensor = self.img_transform(img)
            else:
                im_tensor = to_tensor(img)
            im_tensor = im_tensor.to(self.model.device)
            im_tensor = im_tensor.unsqueeze(0)
            nms = self.prompt(im_tensor)[0][0]
            self._add_to_storage(nms, img, y)
            pass

    def save_objects(self, ds):
        for i in range(len(ds)):
            data = ds[i]
            img = data["raw image"]
            target = data["class_idx"]
            with torch.no_grad():
                self._detect_and_crop(img, target)
        self._compute_normalization_factors()
        self._compute_train_outputs(ds)
        self._compute_bias()
        pass

    def _compute_train_outputs(self, ds):
        targets = []
        outputs = []
        for i in range(len(ds)):
            data = ds[i]
            img = data["raw image"]
            target = data["class_idx"]
            with torch.no_grad():
                out = self.predict(img)
                _, idx = non_max_suppression2(out, 0.0, 0.5, max_det=1, in_place=False)
                detection = out[0, 4:, idx[0]]
                outputs.append(detection)
                targets.append(target)
                pass
        
        self.train_outputs = torch.stack(outputs, dim=0)
        self.train_targets = torch.tensor(targets)
        pass

    def _compute_bias(self):
        self.bias = self.train_outputs.mean(dim=0)

    def _compute_normalization_factors(self):
        classes = sorted(self.embeddings.keys())
        embeddings = torch.stack([self.embedding_aggregator(self.embeddings[i]) for i in classes])
        if self.debias_normalization:
            dot_products = embeddings @ embeddings.T
            self.norm_factors = dot_products.mean(0).sqrt()[:, None]
            self.norm_factors = {i: self.norm_factors[classes.index(i)] for i in classes}
            return
        # self.norm_factors = normalize(embeddings, dim=-1)
        self.norm_factors = embeddings.norm(dim=-1)
        self.norm_factors = {i: self.norm_factors[classes.index(i)] for i in classes}
        pass
    
    def _normalize_embeddings(self, x, y):
        return x / self.norm_factors[y]

    def get_embedding(self, y):
        return self._normalize_embeddings(self.embedding_aggregator(self.embeddings[y]), y)
    
    def get_embeddings(self):
        keys = sorted(self.embeddings.keys())
        return torch.stack([self.get_embedding(k) for k in keys])

    def predict_from_tensor(self, x, use_bias=False):
        x = x.to(self.model.device)
        emb_targets = self.get_embeddings()
        out = self.model(x, emb_targets)
        if use_bias:
            out = out - torch.nn.functional.pad(self.bias, (4, 0), value=0.0)[None, :, None]
        return out
    
    def predict_independent_from_tensor(self, x):
        emb_targets = self.get_embeddings()
        batched_x = x.repeat(len(emb_targets), 1, 1, 1)
        permuted_targets = emb_targets.unsqueeze(0).permute((1, 0, 2))
        return self.model(batched_x, permuted_targets)

    def predict(self, im, use_bias=False):
        x = self.img_transform(im).unsqueeze(0)
        return self.predict_from_tensor(x, use_bias=use_bias)

    def predict_independent(self, im):
        x = self.img_transform(im).unsqueeze(0)
        return self.predict_independent_from_tensor(x)