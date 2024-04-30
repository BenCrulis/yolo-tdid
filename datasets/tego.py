import re
from itertools import chain, product
from pathlib import Path
import numpy as np
from numpy.random import RandomState
import json
import PIL
import PIL.Image
import torch


class TEgO:
    def __init__(self, root, transform=None, max_opened=8000):
        root = Path(root)
        self.root = root

        self.max_opened = max_opened

        self.transform = transform

        images = sorted(list(root.glob("*/Images/*/*/*.jpg")))

        with open(root / "labels_for_training.json", mode="r") as f:
            json_training = json.load(f)
        with open(root / "labels_for_testing.json", mode="r") as f:
            json_testing = json.load(f)

        regex = re.compile("(..)_(..)_(.+)_(?:test|train)_?(.*)")

        labels = []

        wild = np.zeros(len(images), dtype=bool)
        blind = np.zeros(len(images), dtype=bool)
        hand = np.zeros(len(images), dtype=bool)
        torch_lighting = np.zeros(len(images), dtype=bool)
        NIL = np.zeros(len(images), dtype=bool)
        volume_portrait = np.zeros(len(images), dtype=bool)
        testing = np.zeros(len(images), dtype=bool)
        for i in range(len(images)):
            p = images[i]
            annotations = p.parts[-2]
            vanilla_wild = p.parts[-3]
            train_test = p.parts[-5]
            if vanilla_wild == "in-the-wild":
                wild[i] = True
            if train_test == "Testing":
                testing[i] = True

            label_data = json_testing if train_test == "Testing" else json_training
            label = label_data[vanilla_wild][annotations][p.name]
            labels.append(label)

            m = regex.match(annotations)
            if m:
                blind_sighted = m.group(1)
                volume_screen = m.group(2)
                hand_string = m.group(3)
                extra = m.group(4)

                if blind_sighted == "B1":
                    blind[i] = True
                if volume_screen == "PV":
                    volume_portrait[i] = True
                if hand_string == "H":
                    hand[i] = True
                if extra == "NIL":
                    NIL[i] = True
                elif extra == "torch":
                    torch_lighting[i] = True

                pass
            else:
                raise ValueError(f"could not match {annotations}")

            pass

        unique_labels = sorted(set(labels))
        class_index = np.zeros(len(images), dtype=int)

        for i in range(len(images)):
            label = labels[i]
            class_index[i] = unique_labels.index(label)

        self.paths = images

        self.class_idx = class_index
        self.unique_labels = unique_labels
        self.labels = labels
        self.wild = wild
        self.blind = blind
        self.hand = hand
        self.torch = torch_lighting
        self.non_illuminated = NIL
        self.volume_portrait = volume_portrait
        self.testing = testing

        self.opened = {}  # stores opened image files

    def __len__(self):
        return len(self.paths)

    def _clear_cache(self):
        for k, v in self.opened.items():
            v.close()
        self.opened.clear()

    def __getitem__(self, item):
        path = self.paths[item]

        if len(self.opened) > self.max_opened:
            self._clear_cache()  # prevent too many files to be open at the same time

        im = self.opened.get(path, None)
        if im is None:
            try:
                im: PIL.Image.Image = PIL.Image.open(path)
            except OSError as e:
                if e.errno == 24:  # too many open files error
                    self._clear_cache()
                    im: PIL.Image.Image = PIL.Image.open(path)
                else:
                    raise e
            self.opened[path] = im
        transformed = None
        if self.transform:
            transformed = self.transform(im)

        class_name = self.labels[item]
        additional = {k: getattr(self, k)[item].item() for k in ["wild", "blind", "hand", "torch", "non_illuminated",
                                                          "volume_portrait", "testing"]}

        out = {
            "raw image": im,
            "image": transformed,
            "class": class_name,
            "class_idx": self.class_idx[item],
            **additional
        }

        return out


def collate_fn(batch):
    from torch.utils.data import default_collate
    out = {}
    for k in batch[0].keys():
        items = [x[k] for x in batch]
        if "raw" not in k:
            items = default_collate(items)
        if isinstance(items[0], bool):
            items = items.to(dtype=torch.bool)
        out[k] = items

    return out


if __name__ == '__main__':
    from tqdm import tqdm
    tego_ds = TEgO("/mnt/c/data/datasets/TEgO")

    im = tego_ds[6]

    pass