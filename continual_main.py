from pathlib import Path
import argparse

import torch
from torch import nn
from ultralytics import YOLO, YOLOWorld

import avalanche as al
from avalanche.benchmarks.classic import CORe50
from avalanche.evaluation.metrics import loss_metrics, timing_metrics, forgetting_metrics, \
    topk_acc_metrics, confusion_matrix_metrics, accuracy_metrics
from avalanche.logging import InteractiveLogger, WandBLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin

from continual.benchmarks.core50_fs import CORe50_fs, CORe50_fs_bbox
from continual.detect_and_crop import DetectCropSave

from continual.plugins.saved_objects import SavedObjectCountPluginMetric, SeenObjectCountPluginMetric
from continual.plugins.training_time import TrainTime

from utils.whitening import get_whitening_and_coloring_matrices


def parse_args():
    parser = argparse.ArgumentParser(description='Continual learning with CORe50')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--fsCore50', action='store_true', help='Use full size CORe50')

    parser.add_argument('--strategy', default="detectAndCrop", help='One of {naive,ar1,detectAndCrop}')
    parser.add_argument('--wc', action='store_true', help="use whitening and coloring")
    parser.add_argument('--whitening-data', default="coco_train_embeddings.pt", type=Path, help="Whitening data")
    parser.add_argument('--coloring-data', default="gqa_txt_embeddings.pt", type=Path, help="Coloring data")

    parser.add_argument('--yolo-size', default="s", help='One of {s,m,l} default to s')
    parser.add_argument('--store', default="avg", help='One of {avg,first,last,largest,most_likely,exp_avg,exp_weighted_avg} default to avg')
    parser.add_argument('--min-prob', type=float, default=0.01, help='Minimum probability for detection')
    parser.add_argument('--augment', action="store_true", help='Use deterministic data augmentation')
    parser.add_argument('--alpha', type=float, default=None, help='alpha value for the weight update')
    parser.add_argument('--margin', type=int, default=15, help='Margin for cropping')
    parser.add_argument('--class-lvl', action="store_true", help='Use class level instead of object level Core50')
    parser.add_argument('--cpu', action='store_true', help='Use CPU')
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--fast-eval', action='store_true', help='Use fast evaluation')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    use_wandb = args.wandb
    use_full_size_core50 = args.fsCore50
    strategy = args.strategy
    bs = args.bs
    fast_eval = args.fast_eval
    yolo_model_size = args.yolo_size
    store = args.store
    class_lvl = args.class_lvl
    gt_bbox = True

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device", device)

    WC_data = get_whitening_and_coloring_matrices(args, device=device)

    detectAndCropParams = {
        "store": store,
        "min_prob": args.min_prob,
        "margin": args.margin,
        "augmentations": args.augment,
        "alpha": args.alpha,
        "wc_data": WC_data,
    }

    normalize_Core50 = False  # YOLO-World doesn't seem to need input image normalization

    if use_full_size_core50:
        from torchvision.transforms import Compose, ToTensor

        benchmark_fn = CORe50_fs
        if gt_bbox and strategy != "ar1":
            benchmark_fn = CORe50_fs_bbox

        min_obj_size = 30
        if normalize_Core50:
            benchmark = benchmark_fn(object_lvl=not class_lvl, reduced_eval=fast_eval)
        else:
            train_tf = Compose([ToTensor()])
            test_tf = Compose([ToTensor()])
            benchmark = benchmark_fn(train_transform=train_tf,
                                  eval_transform=test_tf,
                                  object_lvl=not class_lvl,
                                  reduced_eval=fast_eval)
        print("fixing corrupted image")
        corrupted_img_path = benchmark.root / "core50_350x350/s3/o43/C_03_43_209.png"
        fixed_img_path: Path = Path("core50_fix/C_03_43_209.png")
        import shutil
        shutil.copy(fixed_img_path, corrupted_img_path)
        pass
    else:
        min_obj_size = 15
        benchmark = CORe50(object_lvl=not class_lvl)

    n_classes = benchmark.n_classes

    loss_fn = torch.nn.CrossEntropyLoss()

    if strategy == "detectAndCrop":
        run_name = f"{strategy}" + f"_yolow-{yolo_model_size}" \
         + ("_fs" if use_full_size_core50 else "") \
         + ("_cl" if class_lvl else "") \
         + f"_store-{store}" \
         + ("_fast" if fast_eval else "")
    else:
        run_name = f"{strategy}_bs-{bs}"

    # loggers = [TextLogger(),]
    loggers = [InteractiveLogger(),]
    if use_wandb:
        loggers.append(WandBLogger(project_name="yolo_tdid", run_name=run_name,
            config={
                "strategy": strategy,
                "n_classes": n_classes,
                "yolo_size": yolo_model_size,
                "class_lvl": class_lvl,
                "fast_eval": fast_eval,
                **detectAndCropParams,
                "bs": bs,
            }
        ))


    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
        forgetting_metrics(experience=True, stream=True),
        *([confusion_matrix_metrics(n_classes)] if n_classes <= 150 else []),
        SavedObjectCountPluginMetric(reset_at="never", emit_at="iteration", mode="train"),
        SeenObjectCountPluginMetric(reset_at="never", emit_at="iteration", mode="train"),
        TrainTime(),
        #LogPlugin.get_instance(),
        loggers=loggers,
        # benchmark=benchmark
    )

    plugins = []

    common_args = dict(train_mb_size=bs,
                        eval_mb_size=bs,
                        plugins=plugins,
                        evaluator=eval_plugin,
                        device=device)

    if strategy == "ar1":
        cl_strategy = al.training.AR1(criterion=loss_fn,
                                    lr=1e-3,
                                    ewc_lambda=0.0,
                                    train_epochs=4,
                                    rm_sz=2000,
                                    freeze_below_layer="lat_features.19.bn",
                                    **common_args)
        cl_strategy.model.end_features[-1] = nn.AdaptiveAvgPool2d((1, 1))
    elif strategy == "detectAndCrop":
        from models.yolow.yolow_adapter import YoloWModel
        from models.yolow.utils import get_yolow_clip, get_clip_encoders
        from continual.detect_and_crop import ModelWrapper
        from torch.jit import script
        model = YoloWModel(model_size=yolo_model_size, scale=21, device=device)

        from typing import List

        class NewConcat(nn.Module):
            """Concatenate a list of tensors along dimension."""

            def __init__(self, dimension: int = 1):
                """Concatenates a list of tensors along a specified dimension."""
                super().__init__()
                self.d: int = dimension

            def forward(self, x: List[torch.Tensor]):
                """Forward pass for the YOLOv8 mask Proto module."""
                return torch.cat(x, self.d)
        from utils.pytorch import replace_module_fn
        from ultralytics.nn.modules import Concat
        # replace_module_fn(model.model, Concat, lambda m: NewConcat(m.d))
        # wrapped_model = ModelWrapper(script(model, example_inputs=[torch.rand(1, 3, 672, 672).to(device), torch.rand(1, 2, 512).to(device)]))

        wrapped_model = ModelWrapper(model)
        img_enc, txt_enc = get_clip_encoders(not normalize_Core50, device=device)
        img_enc.to(device)
        txt_enc.to(device)
        cl_strategy = DetectCropSave(
            model=wrapped_model,
            text_encoder=txt_enc,
            img_encoder=img_enc,
            criterion=loss_fn,
            train_epochs=1,
            eval_every=-1,
            **detectAndCropParams,
            **common_args
        )

    from avalanche.benchmarks.scenarios.deprecated.classification_scenario import (ClassificationExperience,
     ClassificationStream, ClassificationScenario)
    from avalanche.benchmarks.scenarios.generic_scenario import CLStream
    from avalanche.benchmarks.scenarios.dataset_scenario import DatasetExperience

    print("starting continual training")
    for exp_id, experience in enumerate(benchmark.train_stream):
        print("Start of experience ", experience.current_experience)

        cl_strategy.train(experience, shuffle=True)
        print('Training completed')

        print('Computing accuracy on the current test set')
        if fast_eval and not use_full_size_core50:
            old_dataset = benchmark.test_stream[0].dataset
            new_ds = benchmark.test_stream[0].dataset.subset(range(0, len(old_dataset), 20))
            cl_stream = CLStream("test", [DatasetExperience(dataset=new_ds, current_experience=experience.current_experience)])
            eval_stream = ClassificationStream("test", benchmark)
            cl_strategy.eval(eval_stream)
        else:
            cl_strategy.eval(benchmark.test_stream)

    print("end")
