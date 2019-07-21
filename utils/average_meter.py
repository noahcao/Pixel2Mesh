from collections import Iterable

import torch

import numpy as np


# noinspection PyAttributeOutsideInit
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, multiplier=1.0):
        self.multiplier = multiplier
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.cpu().numpy()
        if isinstance(val, Iterable):
            val = np.array(val)
            self.update(np.mean(np.array(val)), n=val.size)
        else:
            self.val = self.multiplier * val
            self.sum += self.multiplier * val * n
            self.count += n
            self.avg = self.sum / self.count if self.count != 0 else 0

    def __str__(self):
        return "%.6f (%.6f)" % (self.val, self.avg)
