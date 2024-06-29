from pathlib import Path
from typing import Any, Optional, Union, Dict, List, Sequence

from torch.utils.data import Subset

from torchvision.transforms import Compose, ToTensor

from datasets.core50_fs import CORe50_350_Dataset
from avalanche.benchmarks.datasets import default_dataset_location

from avalanche.benchmarks.scenarios.deprecated.generic_benchmark_creation import (
    create_generic_benchmark_from_filelists,
)

from avalanche.benchmarks.classic.core50 import normalize, nbatch, scen2dirs
from avalanche.benchmarks import AvalancheDataset, DatasetScenario, benchmark_from_datasets
from avalanche.benchmarks.utils import ConstantSequence, TaskLabels


_default_train_transform = Compose([ToTensor(), normalize])  # no random horizontal flip since we have bounding boxes
_default_eval_transform = Compose([ToTensor(), normalize])


def CORe50_fs(
    *,
    scenario: str = "nicv2_391",
    run: int = 0,
    object_lvl: bool = True,
    train_transform: Optional[Any] = _default_train_transform,
    eval_transform: Optional[Any] = _default_eval_transform,
    dataset_root: Optional[Union[str, Path]] = None,
    reduced_eval=False,
):
    """
    Creates a CL benchmark for CORe50 full size (350x350).

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    This generator can be used to obtain the NI, NC, NIC and NICv2-* scenarios.

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    The task label 0 will be assigned to each experience.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param scenario: CORe50 main scenario. It can be chosen between 'ni', 'nc',
        'nic', 'nicv2_79', 'nicv2_196' or 'nicv2_391.'
    :param run: number of run for the benchmark. Each run defines a different
        ordering. Must be a number between 0 and 9.
    :param object_lvl: True for a 50-way classification at the object level.
        False if you want to use the categories as classes. Default to True.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param dataset_root: Absolute path indicating where to store the dataset
        and related metadata. Defaults to None, which means that the default
        location for
        'core50' will be used.

    :returns: a properly initialized :class:`GenericCLScenario` instance.
    """

    assert 0 <= run <= 9, (
        "Pre-defined run of CORe50 are only 10. Indicate " "a number between 0 and 9."
    )
    assert scenario in nbatch.keys(), (
        "The selected scenario is note "
        "recognized: it should be 'ni', 'nc',"
        "'nic', 'nicv2_79', 'nicv2_196' or "
        "'nicv2_391'."
    )

    if dataset_root is None:
        dataset_root = default_dataset_location("core50")

    # Download the dataset and initialize filelists
    core_data = CORe50_350_Dataset(root=dataset_root, object_level=object_lvl)
    # print("checking dataset")
    # core_data.check_and_fix()  # todo: make sure we are correctly loading images from lists

    root = core_data.root
    bp = "core50_350x350"
    root_img = root / bp

    if object_lvl:
        suffix = "/"
    else:
        suffix = "_cat/"
    filelists_bp = scen2dirs[scenario][:-1] + suffix + "run" + str(run)
    train_failists_paths = []
    for batch_id in range(nbatch[scenario]):
        train_failists_paths.append(
            root
            / filelists_bp
            / ("train_batch_" + str(batch_id).zfill(2) + "_filelist.txt")
        )

    if reduced_eval:
        test_filelist_path = root / "test_filelist_reduced.txt"
    else:
        test_filelist_path = root / filelists_bp / "test_filelist.txt"

    benchmark_obj = create_generic_benchmark_from_filelists(
        root_img,
        train_failists_paths,
        [test_filelist_path],
        task_labels=[0 for _ in range(nbatch[scenario])],
        complete_test_set_only=True,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    setattr(benchmark_obj, "root", root) # add dataset root attribute

    if scenario == "nc":
        n_classes_per_exp = []
        classes_order = []
        for exp in benchmark_obj.train_stream:
            exp_dataset = exp.dataset
            unique_targets = list(
                sorted(set(int(x) for x in exp_dataset.targets))  # type: ignore
            )
            n_classes_per_exp.append(len(unique_targets))
            classes_order.extend(unique_targets)
        setattr(benchmark_obj, "n_classes_per_exp", n_classes_per_exp)
        setattr(benchmark_obj, "classes_order", classes_order)
    setattr(benchmark_obj, "n_classes", 50 if object_lvl else 10)

    return benchmark_obj


def CORe50_fs_bbox(
    *,
    scenario: str = "nicv2_391",
    run: int = 0,
    object_lvl: bool = True,
    train_transform: Optional[Any] = _default_train_transform,
    eval_transform: Optional[Any] = _default_eval_transform,
    dataset_root: Optional[Union[str, Path]] = None,
    reduced_eval=False,
):
    assert 0 <= run <= 9, (
        "Pre-defined run of CORe50 are only 10. Indicate " "a number between 0 and 9."
    )
    assert scenario in nbatch.keys(), (
        "The selected scenario is note "
        "recognized: it should be 'ni', 'nc',"
        "'nic', 'nicv2_79', 'nicv2_196' or "
        "'nicv2_391'."
    )

    if dataset_root is None:
        dataset_root = default_dataset_location("core50")

    # Download the dataset and initialize filelists
    core_data = CORe50_350_Dataset(root=dataset_root, object_level=object_lvl, return_bbox=True)
    # print("checking dataset")
    # core_data.check_and_fix()  # todo: make sure we are correctly loading images from lists

    root = core_data.root
    bp = "core50_350x350"
    root_img = root / bp

    inv_search = {}
    for i, k in enumerate(core_data.paths):
        inv_search[k] = i

    if object_lvl:
        suffix = "/"
    else:
        suffix = "_cat/"
    filelists_bp = scen2dirs[scenario][:-1] + suffix + "run" + str(run)
    train_failists_paths = []
    for batch_id in range(nbatch[scenario]):
        train_failists_paths.append(
            root
            / filelists_bp
            / ("train_batch_" + str(batch_id).zfill(2) + "_filelist.txt")
        )
    
    transform_groups = {
        "train": (train_transform, None),
        "test": (eval_transform, None),
    }

    from avalanche.benchmarks.utils.utils import _init_transform_groups

    train_datasets = []
    for exp_id, train_filelist_path in enumerate(train_failists_paths):
        with open(train_filelist_path, "r") as f:
            idx = []
            for line in f:
                p, target = line.strip().split(" ")
                target = int(target)
                inv_idx = inv_search[p]
                idx.append(inv_idx)
                pass

            transform_gs = _init_transform_groups(
                transform_groups,
                None,
                None,
                "train", # initial transform group
                None, # dataset
            )

            subset = Subset(core_data, idx)
            ds = AvalancheDataset([subset], transform_groups=transform_gs)
            ds.targets_task_labels = TaskLabels(ConstantSequence(0, len(subset)))
            ds.current_experience = exp_id
            train_datasets.append(ds)
            pass
    del idx

    core_data_test = CORe50_350_Dataset(root=dataset_root, object_level=object_lvl, train=False, return_bbox=True)
    inv_search_test = {}
    for i, k in enumerate(core_data_test.paths):
        inv_search_test[k] = i

    if reduced_eval:
        test_filelist_path = root / "test_filelist_reduced.txt"
    else:
        test_filelist_path = root / filelists_bp / "test_filelist.txt"

    with open(test_filelist_path, "r") as f:
        test_idx = []
        for line in f:
            p, target = line.strip().split(" ")
            target = int(target)
            inv_idx = inv_search_test[p]
            test_idx.append(inv_idx)
            pass
    
    transform_gs = _init_transform_groups(
        transform_groups,
        None,
        None,
        "test", # initial transform group
        None, # dataset
    )

    test_subset = Subset(core_data_test, test_idx)
    test_ds = AvalancheDataset([test_subset], transform_groups=transform_gs)

    stream_definitions: Dict[str, Sequence] = {
        "train": train_datasets,
        "test": [test_ds]
    }

    benchmark_obj = benchmark_from_datasets(**stream_definitions)

    setattr(benchmark_obj, "root", root) # add dataset root attribute

    if scenario == "nc":
        n_classes_per_exp = []
        classes_order = []
        for exp in benchmark_obj.train_stream:
            exp_dataset = exp.dataset
            unique_targets = list(
                sorted(set(int(x) for x in exp_dataset.targets))  # type: ignore
            )
            n_classes_per_exp.append(len(unique_targets))
            classes_order.extend(unique_targets)
        setattr(benchmark_obj, "n_classes_per_exp", n_classes_per_exp)
        setattr(benchmark_obj, "classes_order", classes_order)
    setattr(benchmark_obj, "n_classes", 50 if object_lvl else 10)

    return benchmark_obj



if __name__ == "__main__":

    # fs_bench = CORe50_fs()
    fs_bench = CORe50_fs_bbox()

    pass