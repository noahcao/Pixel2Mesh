import re

import torch


checkpoint = torch.load("checkpoints/debug/20190705192654/000001_000001.pt")
pretrained = torch.load("checkpoints/pretrained/network_4.pth")

weights = checkpoint["model"]

for k in weights.keys():
    match = k
    match = re.sub("gcns\.(\d)", "GCN_\\1", match)
    match = re.sub("conv(\d)\.weight", "conv\\1.weight_2", match)
    match = re.sub("conv(\d)\.loop_weight", "conv\\1.weight_1", match)
    match = re.sub("gconv\.weight", "GConv.weight_2", match)
    match = re.sub("gconv\.loop_weight", "GConv.weight_1", match)
    match = re.sub("gconv\.", "GConv.", match)
    if match not in pretrained:
        print(k, match)
    else:
        weights[k] = pretrained[match]
torch.save(checkpoint, "checkpoints/debug/migration/network_4.pt")
