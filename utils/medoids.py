import torch


def cosine_medoid(xs):
    # stack tensors if it is an enumerable
    if isinstance(xs, (list, tuple)):
        xs = torch.stack(xs, dim=0)
    
    # normalize vectors
    xs = torch.nn.functional.normalize(xs, p=2, dim=1)

    # calculate the distance matrix using cosine
    dist = 1 - torch.matmul(xs, xs.t())

    # find the medoid
    medoid = torch.argmin(dist.sum(1))

    return xs[medoid]