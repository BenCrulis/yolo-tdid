import os
from pathlib import Path
import argparse
import random

import csv

import numpy as np

import torch
from torch import nn
from torch.utils.data import Subset, SubsetRandomSampler, SequentialSampler, DataLoader
from ultralytics import YOLO, YOLOWorld
from ultralytics.utils.ops import non_max_suppression

from torch.jit import script

from datasets.tego import TEgO

from models.yolow.yolow_adapter import YoloWModel
from models.yolow.utils import get_yolow_clip, get_clip_encoders

from utils.nms import non_max_suppression2
from utils.pytorch import seed_everything

from tdid.prompts import DetectObjectPrompt, object_in_hand_prompt, main_object_prompt
from tdid.tdid_from_crops import TDIDFromCrops
from tdid.aggregations import average_embeddings


def stratified_sampler(idx, class_idx, k):
    idx = np.array(idx)
    class_idx = np.array(class_idx)
    classes = np.unique(class_idx)
    n_classes = len(classes)
    n = k * n_classes
    idx_per_class = []
    for c in classes:
        idx_c = idx[class_idx == c]
        idx_c = np.random.permutation(idx_c)
        idx_per_class.append(idx_c[:k])
    idx_per_class = np.concatenate(idx_per_class)
    return idx_per_class


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation on TEgO')
    parser.add_argument('-r', '--repeat', type=int, default=1, help='Number of repetitions of the experiment')
    parser.add_argument('-n', type=int, default=10, help='Number of objects to evaluate')
    parser.add_argument('-k', type=int, default=5, help='Number of images per object')
    parser.add_argument('--tego', type=Path, default=None, help='TEgO dataset root')
    parser.add_argument('--yolo-size', default="s", help='One of {s,m,l} default to s')

    # Data selection
    parser.add_argument('--blind', action='store_true', help="Use sighted data")
    parser.add_argument('--clutter', action='store_true', help="Use cluttered data")
    parser.add_argument('--hand', action='store_true', help="Use hand data")

    parser.add_argument('--store', default="avg", help='One of {avg,first,last,largest,most_likely,exp_avg,exp_weighted_avg} default to avg')
    parser.add_argument('--min-prob', type=float, default=0.01, help='Minimum probability for detection')
    parser.add_argument('--margin', type=int, default=15, help='Margin for cropping')
    parser.add_argument('--use-bias', action='store_true', help="Use bias in prediction")
    parser.add_argument('--augment', action='store_true', help="Use augmentations")
    parser.add_argument('--dry', action='store_true', help="dry run (don't save anything)")
    parser.add_argument('--cpu', action='store_true', help='Use CPU')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    return parser.parse_args()


def resolve_tego_path(args):
    if args.tego is None:
        if "TEGO_ROOT" not in os.environ:
            raise ValueError("TEgO dataset root not provided and TEGO_ROOT not set in environment")
        return Path(os.environ["TEGO_ROOT"])
    else:
        return args.tego


def evaluate(tego_root, seed, repeats=1, yolo_model_size="s", margin=5, sighted_data=True, clutter=True, hand=True,
             n_obj=10, k=5, use_bias=False, augment=False, independent_predictions=False, dry=False, force_cpu=False):
    N_CLASSES = 19
    device = torch.device("cuda:0" if torch.cuda.is_available() and not force_cpu else "cpu")
    print("Using device", device)

    run_name = f"yolow-{yolo_model_size}_sighted-{sighted_data}_clutter-{clutter}_hand-{hand}_n-{n_obj}_k-{k}_bias-{use_bias}_augment-{augment}_seed-{seed}"
    
    model = YoloWModel(model_size=yolo_model_size, scale=21, device=device)

    # wrapped_model = ModelWrapper(model)
    img_enc, txt_enc = get_clip_encoders(True, device=device)
    img_enc.to(device)
    txt_enc.to(device)
    img_enc.eval()
    txt_enc.eval()

    tego_ds: TEgO = TEgO(tego_root)

    class_idx = np.unique(tego_ds.class_idx)
    print(f"{len(class_idx)} classes in dataset")

    out_path = Path("results")
    if not dry:
        out_path.mkdir(exist_ok=True, parents=True)
        print(f"will write to {out_path / run_name}.csv")
        file = open(out_path / f"{run_name}.csv", "w")
        csv_writer = csv.writer(file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["repeat", "seed", "hand", "blind", "illuminated", "torch", "volume_portrait", "clutter",
                            "target_class", "predicted_class", "predicted_prob", "correct"] + [f"prob_cls_{i}" for i in range(N_CLASSES)])
        file.flush()

    for it in range(1, repeats+1):
        print(f"iteration {it}/{repeats}")
        iteration_seed = seed + it - 1
        seed_everything(iteration_seed)
        print(f"Using seed {iteration_seed}")

        objects = np.random.permutation(class_idx)[:n_obj]
        objects.sort()
        print(f"Selected {n_obj} objects: {objects}")

        object_mask = np.isin(tego_ds.class_idx, objects)

        if sighted_data:
            sight_mask = ~tego_ds.blind
        else:
            sight_mask = tego_ds.blind
        
        if clutter:
            clutter_mask = tego_ds.wild
        else:
            clutter_mask = ~tego_ds.wild
        
        if hand:
            hand_mask = tego_ds.hand
        else:
            hand_mask = ~tego_ds.hand

        train_mask = (~tego_ds.testing) & sight_mask & clutter_mask & object_mask & hand_mask
        eval_mask = tego_ds.testing & object_mask

        train_idx = stratified_sampler(train_mask.nonzero()[0], tego_ds.class_idx[train_mask], k)

        if len(train_idx) < n_obj * k:
            raise ValueError(f"Could not sample {n_obj * k} images, only {len(train_idx)} available (seed {iteration_seed})")

        tego_train = Subset(tego_ds, train_idx)
        tego_eval = Subset(tego_ds, eval_mask.nonzero()[0])

        tdid = TDIDFromCrops(model, main_object_prompt(model, txt_enc, device), img_enc, average_embeddings,
                             margin=margin,
                             augmentations=augment)

        print("saving objects in model")
        tdid.save_objects(tego_train)

        print("doing inference")

        eval_iterator = range(len(tego_eval))
        using_tqdm = False
        try:
            import tqdm
            eval_iterator = tqdm.tqdm(eval_iterator)
            using_tqdm = True
        except ImportError:
            print("tqdm not available, not using progress bar")

        n_correct = 0
        n_correct_ind = 0
        fraction_correct_ind = -1

        with torch.inference_mode():
            for i in eval_iterator:
                data = tego_eval[i]
                im = data["raw image"]
                test_hand = data["hand"]
                test_sighted = not data["blind"]
                illuminated = not data["non_illuminated"]
                test_torch = data["torch"]
                volume_portrait = data["volume_portrait"]
                test_clutter = data["wild"]

                target = data["class_idx"]
                pred = tdid.predict(im, use_bias=use_bias)
                nms, nms_idx = non_max_suppression2(pred, None, 0.5, max_det=1, in_place=False)
                nms = nms[0]

                predicted_cls = objects[nms[0, 5].long()]
                predicted_prob = nms[0, 4].item()

                all_probs = np.zeros(N_CLASSES)
                all_probs[objects] = pred[0, 4:, nms_idx].squeeze().cpu().numpy()

                if predicted_cls == target:
                    n_correct += 1

                if not dry:
                    csv_writer.writerow([it, iteration_seed, test_hand, test_sighted, illuminated, test_torch, volume_portrait,
                                        test_clutter, target, predicted_cls, predicted_prob, predicted_cls == target] +\
                                            all_probs.tolist())
                
                fraction_correct = n_correct / (i + 1)

                if independent_predictions:
                    pred_ind = tdid.predict_independent(im)
                    nms_ind = torch.cat(non_max_suppression(pred_ind, 0.0, 0.5, max_det=1), dim=0)

                    predicted_cls_ind_idx = nms_ind[:, 4].argmax(dim=0).item()
                    predicted_cls_ind = objects[predicted_cls_ind_idx]
                    predicted_prob_ind = nms_ind[predicted_cls_ind_idx, 4].item()

                    if predicted_cls_ind == target:
                        n_correct_ind += 1

                    fraction_correct_ind = n_correct_ind / (i + 1)

                if not using_tqdm:
                    print(f"({fraction_correct*100:.0f}%) Predicted {predicted_cls} with prob {predicted_prob:.3f} for target {target}")
                else:
                    eval_iterator.set_description(f"accuracy: {round(fraction_correct*100):>3}%, "
                    f"Predicted {predicted_cls:>2} with prob {predicted_prob:.3f} for target {target:>2}")
            pass
        print(f"end of iteration {it}")
        print(f"Accuracy: {fraction_correct*100:.2f}%")
        print()
    if not dry:
        file.flush()
        file.close()
    pass


def main():
    args = parse_args()
    print(args)
    n_repeats = args.repeat
    n_obj = args.n
    k = args.k
    yolo_model_size = args.yolo_size

    sighted_data = not args.blind
    clutter = args.clutter
    hand = args.hand

    margin = args.margin
    store = args.store
    use_bias = args.use_bias
    augment = args.augment
    dry = args.dry

    seed = args.seed
    if seed is None:
        seed = random.randint(0, 1000000)

    independent_predictions = False

    tego_root = resolve_tego_path(args)

    evaluate(
        tego_root,
        repeats=n_repeats,
        seed=seed,
        yolo_model_size=yolo_model_size,
        margin=margin,
        sighted_data=sighted_data,
        clutter=clutter,
        hand=hand,
        n_obj=n_obj,
        k=k,
        use_bias=use_bias,
        augment=augment,
        dry=dry,
        force_cpu=args.cpu)
    print("end")


if __name__ == "__main__":
    main()
