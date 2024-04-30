import os
from pathlib import Path
import argparse
import random

import numpy as np

import torch
from torch import nn
from torch.utils.data import Subset, SubsetRandomSampler, SequentialSampler, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
from ultralytics import YOLO, YOLOWorld
from ultralytics.utils.ops import non_max_suppression

from torch.jit import script

from datasets.tego import TEgO

from models.yolow.yolow_adapter import YoloWModel
from models.yolow.utils import get_yolow_clip, get_clip_encoders
from continual.detect_and_crop import ModelWrapper

from utils.nms import non_max_suppression2
from utils.boxes import crop_from_normalized_bb
from utils.pytorch import pad_to_square

from tdid.prompts import DetectObjectPrompt, object_in_hand_prompt, main_object_prompt
from tdid.tdid_from_crops import TDIDFromCrops, default_transform
from tdid.aggregations import average_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description='Computing TEgO embeddings')
    parser.add_argument('--tego', type=Path, default=None, help='TEgO dataset root')
    parser.add_argument('--yolo-size', default="l", help='One of {s,m,l} default to l')
    parser.add_argument('--margin', type=int, default=15, help='Margin for cropping')
    parser.add_argument('--augment', action='store_true', help="Use augmentations")
    parser.add_argument('--save-crops', action='store_true', help='Store cropped images')
    parser.add_argument('--dry', action='store_true', help="dry run (don't save anything)")
    parser.add_argument('--device', type=str, default="cuda:0", help='choose device (default to cuda:0 if available)')
    return parser.parse_args()


def resolve_tego_path(args):
    if args.tego is None:
        if "TEGO_ROOT" not in os.environ:
            raise ValueError("TEgO dataset root not provided and TEGO_ROOT not set in environment")
        return Path(os.environ["TEGO_ROOT"])
    else:
        return args.tego


def compute_embeddings(tego_root, yolo_model_size="s", margin=5, augment=False, save_crops=False,
                       dry=False, device=None):
    N_CLASSES = 19

    model = YoloWModel(model_size=yolo_model_size, scale=21, device=device)

    # wrapped_model = ModelWrapper(model)
    img_enc, txt_enc = get_clip_encoders(True, device=device)
    img_enc.to(device)
    txt_enc.to(device)
    img_enc.eval()
    txt_enc.eval()

    tego_ds: TEgO = TEgO(tego_root)

    if save_crops:
        crop_dir = Path("tego_crops")

    class_idx = np.unique(tego_ds.class_idx)
    print(f"{len(class_idx)} classes in dataset")

    obj_detector_prompt = main_object_prompt(model, txt_enc, device)

    eval_iterator = range(len(tego_ds))
    using_tqdm = False
    try:
        import tqdm
        eval_iterator = tqdm.tqdm(eval_iterator)
        using_tqdm = True
    except ImportError:
        print("tqdm not available, not using progress bar")

    embeddings = []

    with torch.inference_mode():
        for i in eval_iterator:
            data = tego_ds[i]
            target = data["class_idx"]
            img = data["raw image"]

            im_tensor = default_transform(img)
            im_tensor = im_tensor.to(device)
            im_tensor = im_tensor.unsqueeze(0)
            nms = obj_detector_prompt(im_tensor)[0][0]
            
            cropped = crop_from_normalized_bb(pad_to_square(to_tensor(img), value=0.5), nms[0, :4], margin=margin, square=True)
            cropped = cropped.to(device)
            cropped_squared = pad_to_square(cropped, value=0.5)

            if save_crops:
                crop_path = crop_dir / f"class_{target}" / f"{i}_cropped.png"
                if not crop_path.exists():
                    crop_path.parent.mkdir(exist_ok=True, parents=True)
                    to_pil_image(cropped).save(crop_path)

            batch = [cropped_squared]
            if augment:
                # add augmented versions of the cropped image with rotations and flips
                batch.append(torch.flip(cropped_squared, [2]))
                rotated90 = torch.rot90(cropped_squared, 1, [1, 2])
                batch.append(rotated90)
                rotated90anti = torch.rot90(cropped_squared, -1, [1, 2])
                batch.append(rotated90anti)
            
            batch = torch.stack(batch, dim=0)

            emb = img_enc(batch)
            embeddings.append(emb)
            pass
    
    if not dry:
        embeddings = torch.stack(embeddings, dim=0)
        torch.save(embeddings, "tego_embeddings.pt")
        print("Embeddings saved")
    pass


def main():
    args = parse_args()
    print(args)
    yolo_model_size = args.yolo_size
    margin = args.margin
    augment = args.augment
    save_crops = args.save_crops
    dry = args.dry

    tego_root = resolve_tego_path(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device", device)

    compute_embeddings(
        tego_root,
        yolo_model_size=yolo_model_size,
        margin=margin,
        augment=augment,
        save_crops=save_crops,
        dry=dry,
        device=device)
    print("end")


if __name__ == "__main__":
    main()
