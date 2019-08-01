import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss().cuda()

    def forward(self, outputs, targets):
        labels = targets["labels"]
        loss = self.cross_entropy(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        return loss, {"loss": loss, "acc": correct / total}
