import os
import pickle

import numpy as np
from torch.utils.data import Dataset

word_idx = {'02691156': 0,  # airplane
            '03636649': 1,  # lamp
            '03001627': 2}  # chair

idx_class = {0: 'airplane', 1: 'lamp', 2: 'chair'}


class ShapeNet(Dataset):
    """
    Dataset wrapping images and target meshes for ShapeNet dataset.
    """

    def __init__(self, file_root, file_list_name):
        self.file_root = file_root
        # Read file list
        with open(os.path.join(self.file_root, "meta", file_list_name + ".txt"), "r") as fp:
            self.file_names = fp.read().split("\n")[:-1]

    def __getitem__(self, index):
        label, filename = self.file_names[index].split("_", maxsplit=1)
        with open(os.path.join(self.file_root, "data", label, filename), "rb") as f:
            data = pickle.load(f, encoding="latin1")
        img, pts, normals = data[0].astype(np.float32) / 255.0, data[1][:, :3], data[1][:, 3:]
        img = np.transpose(img, (2, 0, 1))

        return {
            "images": img,
            "points": pts,
            "normals": normals,
            "labels": label,
            "filename": filename
        }

    def __len__(self):
        return len(self.file_names)
