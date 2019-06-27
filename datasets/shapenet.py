import os
import pickle

import numpy as np
import torch.utils.data as data

word_idx = {'02691156': 0,  # airplane
            '03636649': 1,  # lamp
            '03001627': 2}  # chair

idx_class = {0: 'airplane', 1: 'lamp', 2: 'chair'}


class ShapeNet(data.Dataset):
    """
    Dataset wrapping images and target meshes for ShapeNet dataset.
    """

    def __init__(self, file_root, file_list_name):
        self.file_root = file_root
        # Read file list
        with open(os.path.join(file_root, "meta", file_list_name + ".txt"), "r") as fp:
            self.file_names = fp.read().split("\n")[:-1]
        self.file_nums = len(self.file_names)

    def __getitem__(self, index):
        name = os.path.join(self.file_root, self.file_names[index])
        data = pickle.load(open(name, "rb"), encoding='latin1')
        img, pts, normals = data[0].astype('float32') / 255.0, data[1][:, :3], data[1][:, 3:]
        img = np.transpose(img, (2, 0, 1))
        label = word_idx[self.file_names[index].split('_')[0]]

        return {
            "images": img,
            "points": pts,
            "normals": normals,
            "labels": label,
            "filename": self.file_names[index]
        }

    def __len__(self):
        return self.file_nums
