import torch


def recursive_detach(t):
    if isinstance(t, torch.Tensor):
        return t.detach()
    elif isinstance(t, list):
        return [recursive_detach(x) for x in t]
    elif isinstance(t, dict):
        return {k: recursive_detach(v) for k, v in t.items()}
