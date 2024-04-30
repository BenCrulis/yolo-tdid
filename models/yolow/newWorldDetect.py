import torch
from torch import nn
from torch.nn import functional as F

from ultralytics.nn.modules.block import DFL, Proto, ContrastiveHead, BNContrastiveHead
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules import Detect, WorldDetect

from ultralytics.utils.tal import make_anchors


class NewWorldDetect(Detect):
    def __init__(self, nc=80, embed=512, with_bn=False, ch=()):
        """Initialize YOLOv8 detection layer with nc classes and layer channels ch."""
        super().__init__(nc=nc, ch=ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)
        self.export = True

    def forward(self, x, text, apply_sigmoid=False):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            a = self.cv2[i](x[i])
            b = self.cv4[i](self.cv3[i](x[i]), text)
            x[i] = (a, b)

        box, cls = zip(*x)

        box = torch.cat([c.flatten(2) for c in box], 2)
        cls = torch.cat([c.flatten(2) for c in cls], 2)

        # Inference path
        shape = x[0][0].shape  # BCHW

        # todo: fix this so that we don't need to specify image size in advance
        # self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        # self.shape = shape

        # Precompute normalization factor to increase numerical stability
        # See https://github.com/ultralytics/ultralytics/issues/7371
        grid_h = shape[2]
        grid_w = shape[3]
        grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
        norm = self.strides.to(device=box.device) / (self.stride[0].to(device=box.device) * grid_size)
        dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0).to(device=box.device) * norm[:, :2])
        
        if apply_sigmoid:
            y = torch.cat((dbox, torch.sigmoid(cls)), 1)
        else:
            y = torch.cat((dbox, cls), 1)
        return y
    

def worldDetect_to_newWorldDetect(worldDetect: WorldDetect):
    c3 = min(worldDetect.nc, 100)
    # ch = (c3,)
    ch = tuple(c3 for _ in range(worldDetect.nl))
    newWorldDetect = NewWorldDetect(nc=worldDetect.nc, ch=ch)
    newWorldDetect.cv2 = worldDetect.cv2
    newWorldDetect.cv3 = worldDetect.cv3
    newWorldDetect.cv4 = worldDetect.cv4
    newWorldDetect.dfl = worldDetect.dfl
    newWorldDetect.f = worldDetect.f
    newWorldDetect.nl = worldDetect.nl
    newWorldDetect.stride = worldDetect.stride
    newWorldDetect.strides = worldDetect.strides
    newWorldDetect.anchors = worldDetect.anchors
    newWorldDetect.i = worldDetect.i
    return newWorldDetect
