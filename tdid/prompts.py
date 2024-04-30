import torch

from utils.nms import non_max_suppression2


class DetectObjectPrompt():
    def __init__(self, model, txt_encoder, prompt_classes, device=None):
        self.model = model
        self.txt_encoder = txt_encoder
        self.prompt_classes = prompt_classes

        with torch.inference_mode():
            self.feats = self.txt_encoder(prompt_classes, device=device)
            self.feats /= self.feats.norm(p=2, dim=-1, keepdim=True)
        
    def __call__(self, x):
        feats = self.feats.to(x.device)
        if len(x.shape) == 4:
            feats = feats.unsqueeze(0)
        out = self.model(x, feats)
        nms, idx = non_max_suppression2(out[:, list(range(5)), :], # select the first class (target), other classes are ignored
                                  conf_thres=0.0,
                                  iou_thres=0.5,
                                  classes=None,
                                  agnostic=False,
                                  multi_label=False,
                                  max_det=1,
                                  in_place=False)
                                  
        return nms, idx


def object_in_hand_prompt(model, txt_encoder) -> DetectObjectPrompt:
    return DetectObjectPrompt(model, txt_encoder, ["object held in hand", "hand"])


def main_object_prompt(model, txt_encoder, device=None) -> DetectObjectPrompt:
    return DetectObjectPrompt(model, txt_encoder, ["main object"], device=device)
