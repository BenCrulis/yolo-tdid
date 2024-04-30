
def crop_from_normalized_bb(x, bb, margin=0, square=False):
    x1, y1, x2, y2 = (bb * x.shape[-1]).tolist()
    if square:
        size = max(x2 - x1, y2 - y1)
        x1 = x1 + (x2 - x1 - size) / 2
        y1 = y1 + (y2 - y1 - size) / 2
        x2 = x1 + size
        y2 = y1 + size

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(x.shape[-1], x2 + margin)
    y2 = min(x.shape[-2], y2 + margin)
    return x[..., int(y1):int(y2), int(x1):int(x2)]
