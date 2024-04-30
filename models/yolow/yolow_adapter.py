import torch
from torch import nn

from ultralytics.models.yolo import YOLO
from ultralytics.nn.modules import WorldDetect, C2fAttn, ImagePoolingAttn

from models.yolow.newWorldDetect import NewWorldDetect, worldDetect_to_newWorldDetect
from utils.pytorch import replace_module_fn


class YoloWModel(nn.Module):
    def __init__(self, model_size="s", scale=11, embed=None, device=None) -> None:
        super().__init__()

        name_template = "yolov8{}-worldv2.pt"
        model_name = name_template.format(model_size)

        # Initialize a YOLO-World model
        yolo_model = YOLO(model_name).to(device=device)
        yolo_model.model.eval()

        self.scale = scale
        with torch.inference_mode():
            out = yolo_model(torch.ones((1, 3, 32*scale, 32*scale), device=device))

        # yolo_model.set_classes(["object", "bus"])

        self.model = yolo_model.model.model
        self.model.eval()
        self.embed = embed
        self.save = yolo_model.model.save
        self.device = device

        replace_module_fn(self.model, WorldDetect, worldDetect_to_newWorldDetect)

    def forward(self, x, txt_feats, apply_sigmoid=True):
        txt_feats = txt_feats.to(device=x.device, dtype=x.dtype)
        if len(txt_feats) != len(x):
            txt_feats = txt_feats.repeat(len(x), 1, 1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect) or isinstance(m, NewWorldDetect):
                x = m(x, ori_txt_feats, apply_sigmoid=apply_sigmoid)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if self.embed and m.i in self.embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(self.embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x
