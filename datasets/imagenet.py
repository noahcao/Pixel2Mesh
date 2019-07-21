import os

import cv2
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms


class ImageNet(Dataset):

    def __init__(self, root_dir, split="train"):
        self.image_dir = os.path.join(root_dir, split)
        self.images = []
        self.labels = []
        with open(os.path.join(root_dir, "meta", split + ".txt"), "r") as f:
            for line in f.readlines():
                image, label = line.strip().split()
                self.images.append(image)
                self.labels.append(int(label))

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.image_dir, self.images[index]))[:, :, ::-1]
        image = np.transpose(image, (2, 0, 1)).astype(np.float) / 255.0
        return {
            "images": image,
            "targets": self.labels[index]
        }
