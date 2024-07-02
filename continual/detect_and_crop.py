from typing import Callable, Optional, Sequence, List, Union
import time
import math
import torch
from torch.nn.functional import normalize
from torch.nn.parameter import Parameter

from torch import sigmoid, log
from torch import nn
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from avalanche.models.packnet import PackNetModel, PackNetModule, PackNetPlugin

import torchvision
from torchvision.ops import nms

from avalanche.models.pnn import PNN
from avalanche.training.plugins.evaluation import (
    default_evaluator,
    default_loggers,
)
from avalanche.training.plugins import (
    SupervisedPlugin,
    CWRStarPlugin,
    ReplayPlugin,
    GenerativeReplayPlugin,
    TrainGeneratorAfterExpPlugin,
    GDumbPlugin,
    LwFPlugin,
    AGEMPlugin,
    GEMPlugin,
    EWCPlugin,
    EvaluationPlugin,
    SynapticIntelligencePlugin,
    CoPEPlugin,
    GSS_greedyPlugin,
    LFLPlugin,
    MASPlugin,
    BiCPlugin,
    MIRPlugin,
    FromScratchTrainingPlugin,
)
from avalanche.training.templates.base import BaseTemplate
from avalanche.training.templates import SupervisedTemplate
from avalanche.evaluation.metrics import loss_metrics
from avalanche.models.generator import MlpVAE, VAE_loss
from avalanche.models.expert_gate import AE_loss
from avalanche.logging import InteractiveLogger
from avalanche.training.templates.strategy_mixin_protocol import CriterionType

from ultralytics.utils.ops import non_max_suppression, xywh2xyxy

from utils.nms import non_max_suppression2
from utils.pytorch import freeze_model, ensure_int
from utils.medoids import cosine_medoid
from utils.boxes import crop_from_normalized_bb

from torchvision.transforms.functional import to_tensor, to_pil_image


from utils.debug import label_img_with_box


class ModelWrapper(nn.Module):
    def __init__(self, model: Module):
        super().__init__()
        self.model = model
    
    def __call__(self, x, txt_feats, task_id=None):
        # scale = 11
        # scale = 21
        scale = self.model.scale
        rescaled_x = nn.functional.interpolate(x, size=(32*scale, 32*scale), mode='bilinear', align_corners=False)
        return self.model(rescaled_x, txt_feats)


def _apply_sigmoid_to_yolo_output(x):
    return torch.cat([x[..., :4], sigmoid(x[..., 4:])], dim=-1)


def _apply_log_to_yolo_output(x):
    return torch.cat([x[..., :4], log(x[..., 4:])], dim=-1)


def expand_log_prob_tensor(logprobs, known_classes, used_classes, filler):
    n_classes = max(known_classes) + 1
    expanded_logprobs = torch.full((logprobs.shape[0], n_classes), filler, device=logprobs.device)
    expanded_logprobs[:, list(used_classes)] = logprobs
    return expanded_logprobs


class DetectCropSave(SupervisedTemplate):
    """Detect object, crop and accumulate a description embedding.
    """

    def __init__(
        self,
        *,
        model: Module,
        text_encoder: Module,
        img_encoder: Module,
        store="avg",
        min_obj_size=25,
        min_prob=0.0,
        margin=5,
        augmentations=False,
        wc_data=None,
        alpha=None,
        criterion: CriterionType = CrossEntropyLoss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        **base_kwargs
    ):
        """
        Creates an instance of the Detect And Crop strategy.

        :param model: The model.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        
        # freeze model parameters and set model to eval mode just in case
        freeze_model(model)
        model.eval()

        super().__init__(
            model=model,
            optimizer=None,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )

        # both should have __call__ method
        self.text_encoder = text_encoder.to(device=device)
        self.img_encoder = img_encoder.to(device=device)
        self.text_encoder.eval()
        self.img_encoder.eval()
        freeze_model(self.text_encoder)
        freeze_model(self.img_encoder)

        self.store = store
        self.min_prob = min_prob
        self.min_obj_size = min_obj_size
        self.margin = margin
        self.crop_to_square = True
        self.augmentations = augmentations
        self.wc_data = wc_data
        self.alpha = alpha

        with torch.inference_mode():
            self.hand_obj_feats = self.text_encoder(["hand", "object held in hand"], device=self.device)
            self.hand_obj_feats /= self.hand_obj_feats.norm(p=2, dim=-1, keepdim=True)

        self.eps = 1e-8

        # dynamic members
        self.feats = {}
        self.known_classes = set()
        self.weights = {}
    

    def _detect_object(self, x):
        if len(self.mbatch) == 2:
            out = self.model(x, self.hand_obj_feats)
            nms, idx = non_max_suppression2(out[:, list(range(4)) + [5], :], # select the right class
                                    conf_thres=0.0,
                                    iou_thres=0.5,
                                    classes=None,
                                    agnostic=False,
                                    multi_label=False,
                                    max_det=1,
                                    in_place=False)
            most_likely_detects = torch.take_along_dim(out, idx.unsqueeze(-1).unsqueeze(-1), dim=-1).squeeze(-1)
            # im = label_img_with_box(to_pil_image(x[0]), nms[0][0,:4]*350, f"{nms[0][0, 4].item():.3f}")
            # im.save("test.jpg")

            # bb format is normalized x1, y1, x2, y2

            for i in range(len(nms)):
                nms_i = nms[i]
                if nms_i.shape[0] > 0:
                    self._update_object_representation(x[i], self.mb_y[i], nms_i)
        elif len(self.mbatch) == 3:  # minibatch also has the bounding boxes
            gt_bboxes = self.mbatch[2]  # a priori, same format as NMS2 output
            fake_nms = gt_bboxes / self.mb_x.shape[-1] # normalize to [0, 1] using image width
            fake_nms = torch.cat([fake_nms, torch.ones((fake_nms.shape[0], 2), device=self.device)], dim=-1)
            fake_nms = fake_nms.unsqueeze(1)

            for i in range(len(fake_nms)):
                fake_nms_i = fake_nms[i]
                if fake_nms_i.shape[0] > 0:
                    self._update_object_representation(x[i], self.mb_y[i], fake_nms_i)

            pass
        pass

    def _classify(self, x, known_classes):
        saved_object_feats, mapping = self._get_all_object_representations()
        saved_object_feats = saved_object_feats / saved_object_feats.norm(p=2, dim=-1, keepdim=True)
        classes = sorted(self.feats.keys())
        weights = torch.tensor([self.weights[y] for y in classes], device=self.device)
        weights = weights.shape[0] * weights / weights.sum()
        saved_object_feats *= weights.unsqueeze(-1)

        # normalize embeddings
        # saved_object_feats = saved_object_feats / saved_object_feats.norm(dim=-1, keepdim=True)

        out = self.model(x, saved_object_feats.unsqueeze(0))
        bef_nms = time.time()
        nms, idx = non_max_suppression2(out,
                                  conf_thres=0.0,
                                  iou_thres=0.5,
                                  classes=None,
                                  agnostic=False,
                                  multi_label=False,
                                  max_det=1,
                                  in_place=False)
        nms_elapsed = time.time() - bef_nms
        most_likely_detects = torch.take_along_dim(out, idx.unsqueeze(-1).unsqueeze(-1), dim=-1).squeeze(-1)

        self.mb_detects = most_likely_detects

        probs = most_likely_detects[:, 4:]
        idx = probs.max(1).indices

        classes = mapping[idx]

        expanded = expand_log_prob_tensor(
            log(probs),
            sorted(known_classes),
            sorted(self.feats.keys()),
            math.log(self.eps))

        self.mb_output = expanded
        pass
        

    def _get_object_representation(self, y):
        if self.store == "avg":
            n, s = self.feats[y]
            return s / n
        elif self.store == "exp_avg":
            feat = self.feats.get(y, None)
            if feat is None:
                return None
            return sum([x[1] for x in feat.values()]) / len(feat)
        elif self.store == "exp_weighted_avg":
            feat = self.feats.get(y, None)
            if feat is None:
                return None
            return sum([x[1] for x in feat.values()]) / sum([x[0] for x in feat.values()])
        elif self.store == "exp_medoid":
            feat = self.feats.get(y, None)
            if feat is None:
                return None
            keys = sorted(feat.keys())
            return cosine_medoid([feat[k][1] for k in keys]) # better to preserve the order
        elif self.store in ("first", "last"):
            return self.feats[y]
        elif self.store == "most_likely":
            return self.feats[y][1]
        else:
            raise NotImplementedError(f"Unknown store method: {self.store}")
        return self.feats[y]
    
    def _get_all_object_representations(self):
        classes = sorted(self.feats.keys())
        if len(classes) <= 0:
            return None, None
        return (torch.stack([self._get_object_representation(y) for y in classes], dim=0),
                torch.tensor(classes, device=self.device))

    def _process_crop(self, cropped):
        batch = [cropped]
        if self.augmentations:
            batch.append(torch.flip(cropped, [2]))
            rotated90 = torch.rot90(cropped, 1, [-2, -1])
            batch.append(rotated90)
            rotated90anti = torch.rot90(cropped, -1, [-2, -1])
            batch.append(rotated90anti)
        embs = [self._compute_embedding(x)[0] for x in batch]
        return torch.stack(embs, dim=0)

    def _compute_embedding(self, x):
        emb = self.img_encoder(x)
        emb = normalize(emb, dim=-1)
        if self.wc_data is not None:
            W_w, W_c, bias_w, bias_c = self.wc_data
            emb = (emb - bias_w) @ (W_w @ W_c)
            emb = emb + bias_c
        return emb

    def _update_object_representation(self, x, y, nms):
        y = ensure_int(y)
        feat = self.feats.get(y, None)

        # discard object if detection probability is too low
        detection_prob = nms[0, 4].item()
        if detection_prob < self.min_prob:
            return

        cropped_x = crop_from_normalized_bb(x, nms[0, :4], self.margin, square=self.crop_to_square)

        # discard object if it is too small
        if cropped_x.shape[-1] < self.min_obj_size or cropped_x.shape[-2] < self.min_obj_size:
            return

        # add weight to the object class
        self.weights[y] = self._mean_weight()
        
        if self.store == "first":
            if feat is None:
                feat = self._process_crop(cropped_x.unsqueeze(0))[0]
                self.feats[y] = feat
        elif self.store == "last":
            feat = self._process_crop(cropped_x.unsqueeze(0))[0]
            self.feats[y] = feat
        elif self.store == "avg":
            if feat is None:
                feat = (0, torch.zeros((512,), device=self.device))
                self.feats[y] = feat
            n, s = feat
            # s += self.img_encoder(cropped_x.unsqueeze(0))[0]
            processed = self._process_crop(cropped_x.unsqueeze(0))
            s += processed.sum(0)
            n += processed.shape[0]
            self.feats[y] = (n, s)
        elif self.store in ("exp_avg", "exp_weighted_avg", "exp_medoid"):
            exp_id = self.experience.current_experience
            if feat is None:
                feat = {exp_id: (0.0, torch.zeros((512,), device=self.device))}
                self.feats[y] = feat
            exp_data = feat.get(exp_id, None)
            if exp_data is None:
                exp_data = (0.0, torch.zeros((512,), device=self.device))
                feat[exp_id] = exp_data
            prob, emb = exp_data

            if detection_prob > prob:
                new_exp_data = (detection_prob, self._process_crop(cropped_x.unsqueeze(0))[0])
                feat[exp_id] = new_exp_data
            pass
        elif self.store == "most_likely":
            if feat is None:
                feat = (0.0, torch.zeros((512,), device=self.device))
                self.feats[y] = feat
            prob, s = feat
            if detection_prob > prob:
                feat = (detection_prob, self._process_crop(cropped_x.unsqueeze(0))[0])
                self.feats[y] = feat
        else:
            raise ValueError(f"Unknown store method: {self.store}")

        pass

    def _before_training_exp(self, **kwargs):
        """Setup to train on a single experience."""
        # Data Adaptation (e.g. add new samples/data augmentation)
        self._before_train_dataset_adaptation(**kwargs)
        self.train_dataset_adaptation(**kwargs)
        self._after_train_dataset_adaptation(**kwargs)

        self.make_train_dataloader(**kwargs)

    def _update_known_classes(self):
        self.known_classes = self.known_classes.union(self.mb_y.unique().tolist())
        pass

    def _mean_weight(self):
        if len(self.weights) == 0:
            return 1.0
        return sum([x for x in self.weights.values()]) / len(self.weights)
    
    def _update_weights(self):
        probs = self.mb_detects[:, 4:]

        mapping = sorted(self.feats.keys())

        for i in range(len(self.mb_y)):
            y = self.mb_y[i].item()
            prob = probs[i]
            predicted_cls = mapping[prob.argmax().item()]
            if y != predicted_cls:
                self.weights[predicted_cls] = (1 - self.alpha) * self.weights[predicted_cls]
            pass

    def forward(self, training=False):
        """Compute the model's output given the current mini-batch."""

        # detect, crop and accumulate a description embedding for the objects (if detected)
        # self._detect_object(self.mb_x)

        known_classes = sorted(set(self.known_classes) | set(self.mb_y.unique().tolist()))

        # classify the objects
        self._classify(self.mb_x, known_classes)
        # self.mb_task_id returns the current task id
        # return self.model(self.mb_x, torch.ones((len(self.known_classes), 512), device=self.device))

        if training and self.alpha is not None:
            self._update_weights()

        return self.mb_output

    def criterion(self):
        return self._criterion(self.mb_output, self.mb_y)
    
    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.loss = self._make_empty_loss()

            try:
                # Forward
                with torch.inference_mode(True):
                    bef_updating_classes = time.time()
                    self._update_known_classes()
                    known_classes_elapsed = time.time() - bef_updating_classes

                    self.model.eval()

                    self._before_update(**kwargs)
                    bef_update_objects = time.time()
                    self._detect_object(self.mb_x)
                    aft_update_objects = time.time()
                    self.train_time = aft_update_objects - bef_update_objects
                    self._after_update(**kwargs)
                    
                    can_infer = len(self.feats) > 0

                    self._before_forward(**kwargs)

                    if can_infer:
                        self.forward(training=True)
                    else:
                        self.mb_output = torch.zeros((1, 1), device=self.device)

                    self._after_forward(**kwargs)

                # Loss
                if can_infer:
                    self.loss += self.criterion()
                    self._after_training_iteration(**kwargs)
                pass
            except OSError as e:
                print(f"Error: {e}")
                pass
        pass

            

    