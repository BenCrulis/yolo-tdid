import glob
import os
import re
import pickle as pkl
import dill
from pathlib import Path
from typing import List, Optional, Tuple, Union
from warnings import warn

import torch

from torchvision.datasets.folder import default_loader
from torchvision.transforms import ToTensor

from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.datasets.downloadable_dataset import DownloadableDataset
from avalanche.benchmarks.datasets.core50 import core50_data


data = [
        (
            "core50_350x350.zip",
            "http://bias.csr.unibo.it/maltoni/download/core50/core50_350x350.zip",
            "e304258739d6cd4b47e19adfa08e7571",
        ),
        (
            "bbox.zip",
            "https://vlomonaco.github.io/core50/data/bbox.zip",
            "86065aac7fd33dc90b917fc879b2e34a",
        ),
        (
            "test_filelist_reduced.txt",
            "https://vlomonaco.github.io/core50/data/test_filelist_reduced.txt",
            "5a185daf7ec9c706f667309b6df9137d"
        )
    ]
data += core50_data.data


class CORe50_350_Dataset(DownloadableDataset):
    """CORe50 Pytorch Dataset"""

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        *,
        train=True,
        transform=None,
        target_transform=None,
        loader=default_loader,
        download=True,
        object_level=True,
        return_bbox=False,
    ):
        """Creates an instance of the full-size (350x350) CORe50 dataset.

        :param root: root for the datasets data. Defaults to None, which
            means that the default location for 'core50' will be used.
        :param train: train or test split.
        :param transform: eventual transformations to be applied.
        :param target_transform: eventual transformation to be applied to the
            targets.
        :param loader: the procedure to load the instance from the storage.
        :param download: boolean to automatically download data. Default to
            True.
        :param object_level: if the classification is objects based or
            category based: 50 or 10 way classification problem. Default to True
            (50-way object classification problem)
        """

        if root is None:
            root = default_dataset_location("core50")

        super(CORe50_350_Dataset, self).__init__(root, download=download, verbose=True)

        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.object_level = object_level
        self.return_bbox = return_bbox

        # any scenario and run is good here since we want just to load the
        # train images and targets with no particular order
        self._scen = "ni"
        self._run = 0
        self._nbatch = 8

        # Download the dataset and initialize metadata
        self._load_dataset()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target
                class.
        """

        target = self.targets[index]
        bp = "core50_350x350"

        try:
            img = self.loader(str(self.root / bp / self.paths[index]))
        except OSError as e:
            warn(f"Error loading image {index}: {e}")
            return None, None
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_bbox:
            return img, target, torch.tensor(self.bbox[index])
        return img, target

    def __len__(self):
        return len(self.targets)

    def check_and_fix(self):
        i = 0
        while i < len(self):
            try:
                self[i]
            except OSError as e:
                warn(f"Error loading image {i}: {e}")
                self.paths.pop(i)
                self.targets.pop(i)
                i -= 1
            i += 1

    def _download_dataset(self) -> None:
        data2download: List[Tuple[str, str, str]] = data

        for name in data2download:
            if self.verbose:
                print("Downloading " + name[1] + "...")
            file = self._download_file(name[1], name[0], name[2])
            if name[1].endswith(".zip") and not (self.root / name[0]).with_suffix("").exists():
                if self.verbose:
                    print(f"Extracting {name[0]}...")
                extract_root = self._extract_archive(file)
                if self.verbose:
                    print("Extraction completed!")

    def _load_metadata(self) -> bool:
        bp = "core50_350x350"

        if not (self.root / bp).exists():
            return False

        if not (self.root / "batches_filelists").exists():
            return False
        
        if not (self.root / "bbox").exists():
            return False

        if not (self.root / "test_filelist_reduced.txt").exists():
            return False

        with open(self.root / "paths.pkl", "rb") as f:
            self.train_test_paths = pkl.load(f)

        if self.verbose:
            print("Loading labels...")
        with open(self.root / "labels.pkl", "rb") as f:
            self.all_targets = pkl.load(f)
            self.train_test_targets = []
            for i in range(self._nbatch + 1):
                self.train_test_targets += self.all_targets[self._scen][self._run][i]

        if self.verbose:
            print("Loading LUP...")
        with open(self.root / "LUP.pkl", "rb") as f:
            self.LUP = pkl.load(f)

        if self.verbose:
            print("Loading labels names...")
        with open(self.root / "labels2names.pkl", "rb") as f:
            self.labels2names = pkl.load(f)

        # BBOX
        if self.verbose:
            print("Loading bbox...")
        bbox_path: Path = self.root / "bbox"
        bbox_files = list(bbox_path.glob("*/*.txt"))
        self.bbox_files = bbox_files

        bbox_raw = {}

        for f in bbox_files:
            with open(f, "r") as file:
                lines = file.readlines()

                seq_bbox = []

                for line in lines:
                    line = line.split(" ")
                    x1, y1, x2, y2 = map(int, line[1:])
                    seq_bbox.append((x1, y1, x2, y2))
                    pass
                seq = f.parent.name
                obj = f.stem.split("_")[1].split(".")[0]
                bbox_raw[(seq, obj)] = seq_bbox

        self.bbox_train_test = []
        
        path_re = re.compile(r"(s\d+)/(o\d+)/C_.+_.+_(\d+)\.png")
        for p in self.train_test_paths:
            m: re.Match = path_re.match(p)
            if m:
                scene, obj, i = m.groups()
                i = int(i)
                self.bbox_train_test.append(bbox_raw[(scene, obj)][i])
            else:
                raise ValueError(f"Invalid path: {p}")

        # PATHS
        self.idx_list = []
        if self.train:
            for i in range(self._nbatch):
                self.idx_list += self.LUP[self._scen][self._run][i]
        else:
            self.idx_list = self.LUP[self._scen][self._run][-1]

        self.paths = []
        self.targets = []
        self.bbox = []

        for idx in self.idx_list:
            self.paths.append(self.train_test_paths[idx])
            self.bbox.append(self.bbox_train_test[idx])
            div = 1
            if not self.object_level:
                div = 5
            self.targets.append(self.train_test_targets[idx] // div)

        with open(self.root / "labels2names.pkl", "rb") as f:
            self.labels2names = pkl.load(f)

        if not (self.root / "NIC_v2_79_cat").exists():
            self._create_cat_filelists()

        return True

    def _download_error_message(self) -> str:
        all_urls = [name_url[1] for name_url in data]

        base_msg = (
            "[CORe50] Error downloading the dataset!\n"
            "You should download data manually using the following links:\n"
        )

        for url in all_urls:
            base_msg += url
            base_msg += "\n"

        base_msg += "and place these files in " + str(self.root)

        return base_msg

    def _create_cat_filelists(self):
        """Generates corresponding filelists with category-wise labels. The
        default one are based on the object-level labels from 0 to 49."""

        for k, v in core50_data.scen2dirs.items():
            orig_root_path = os.path.join(self.root, v)
            root_path = os.path.join(self.root, v[:-1] + "_cat")
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            for run in range(10):
                cur_path = os.path.join(root_path, "run" + str(run))
                orig_cur_path = os.path.join(orig_root_path, "run" + str(run))
                if not os.path.exists(cur_path):
                    os.makedirs(cur_path)
                for file in glob.glob(os.path.join(orig_cur_path, "*.txt")):
                    o_filename = file
                    _, d_filename = os.path.split(o_filename)
                    orig_f = open(o_filename, "r")
                    dst_f = open(os.path.join(cur_path, d_filename), "w")
                    for line in orig_f:
                        path, label = line.split(" ")
                        new_label = self._objlab2cat(int(label), k, run)
                        dst_f.write(path + " " + str(new_label) + "\n")
                    orig_f.close()
                    dst_f.close()

    def _objlab2cat(self, label, scen, run):
        """Mapping an object label into its corresponding category label
        based on the scenario."""

        if scen == "nc":
            return core50_data.name2cat[self.labels2names["nc"][run][label][:-1]]
        else:
            return int(label) // 5


if __name__ == "__main__":
    print("executing core50_fs.py from datasets")
    ds = CORe50_350_Dataset(download=False, train=True, object_level=True, return_bbox=True)

    im = ds[0]

    print(im)
    pass