import json
import os
import pickle

import numpy as np
import torch
from PIL import Image
from skimage import io, transform
from torch.utils.data.dataloader import default_collate

import config
from datasets.base_dataset import BaseDataset


class ShapeNet(BaseDataset):
    """
    Dataset wrapping images and target meshes for ShapeNet dataset.
    """

    def __init__(self, file_root, file_list_name, mesh_pos, normalization, shapenet_options):
        super().__init__()
        self.file_root = file_root
        with open(os.path.join(self.file_root, "meta", "shapenet.json"), "r") as fp:
            self.labels_map = sorted(list(json.load(fp).keys()))
        self.labels_map = {k: i for i, k in enumerate(self.labels_map)}
        # Read file list
        with open(os.path.join(self.file_root, "meta", file_list_name + ".txt"), "r") as fp:
            self.file_names = fp.read().split("\n")[:-1]
        self.tensorflow = "_tf" in file_list_name # tensorflow version of data
        self.normalization = normalization
        self.mesh_pos = mesh_pos
        self.resize_with_constant_border = shapenet_options.resize_with_constant_border

    def __getitem__(self, index):
        if self.tensorflow:
            filename = self.file_names[index][17:]
            label = filename.split("/", maxsplit=1)[0]
            pkl_path = os.path.join(self.file_root, "data_tf", filename)
            img_path = pkl_path[:-4] + ".png"
            with open(pkl_path) as f:
                data = pickle.load(open(pkl_path, 'rb'), encoding="latin1")
            pts, normals = data[:, :3], data[:, 3:]
            img = io.imread(img_path)
            img[np.where(img[:, :, 3] == 0)] = 255
            if self.resize_with_constant_border:
                img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE),
                                       mode='constant', anti_aliasing=False)  # to match behavior of old versions
            else:
                img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
            img = img[:, :, :3].astype(np.float32)
        else:
            label, filename = self.file_names[index].split("_", maxsplit=1)
            with open(os.path.join(self.file_root, "data", label, filename), "rb") as f:
                data = pickle.load(f, encoding="latin1")
            img, pts, normals = data[0].astype(np.float32) / 255.0, data[1][:, :3], data[1][:, 3:]

        pts -= np.array(self.mesh_pos)
        assert pts.shape[0] == normals.shape[0]
        length = pts.shape[0]

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        img_normalized = self.normalize_img(img) if self.normalization else img

        return {
            "images": img_normalized,
            "images_orig": img,
            "points": pts,
            "normals": normals,
            "labels": self.labels_map[label],
            "filename": filename,
            "length": length
        }

    def __len__(self):
        return len(self.file_names)


class ShapeNetImageFolder(BaseDataset):

    def __init__(self, folder, normalization, shapenet_options):
        super().__init__()
        self.normalization = normalization
        self.resize_with_constant_border = shapenet_options.resize_with_constant_border
        self.file_list = []
        for fl in os.listdir(folder):
            file_path = os.path.join(folder, fl)
            # check image before hand
            try:
                if file_path.endswith(".gif"):
                    raise ValueError("gif's are results. Not acceptable")
                Image.open(file_path)
                self.file_list.append(file_path)
            except (IOError, ValueError):
                print("=> Ignoring %s because it's not a valid image" % file_path)

    def __getitem__(self, item):
        img_path = self.file_list[item]
        img = io.imread(img_path)

        if img.shape[2] > 3:  # has alpha channel
            img[np.where(img[:, :, 3] == 0)] = 255

        if self.resize_with_constant_border:
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE),
                                   mode='constant', anti_aliasing=False)
        else:
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        img = img[:, :, :3].astype(np.float32)

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        img_normalized = self.normalize_img(img) if self.normalization else img

        return {
            "images": img_normalized,
            "images_orig": img,
            "filepath": self.file_list[item]
        }

    def __len__(self):
        return len(self.file_list)


def get_shapenet_collate(num_points):
    """
    :param num_points: This option will not be activated when batch size = 1
    :return: shapenet_collate function
    """
    def shapenet_collate(batch):
        if len(batch) > 1:
            all_equal = True
            for t in batch:
                if t["length"] != batch[0]["length"]:
                    all_equal = False
                    break
            points_orig, normals_orig = [], []
            if not all_equal:
                for t in batch:
                    pts, normal = t["points"], t["normals"]
                    length = pts.shape[0]
                    choices = np.resize(np.random.permutation(length), num_points)
                    t["points"], t["normals"] = pts[choices], normal[choices]
                    points_orig.append(torch.from_numpy(pts))
                    normals_orig.append(torch.from_numpy(normal))
                ret = default_collate(batch)
                ret["points_orig"] = points_orig
                ret["normals_orig"] = normals_orig
                return ret
        ret = default_collate(batch)
        ret["points_orig"] = ret["points"]
        ret["normals_orig"] = ret["normals"]
        return ret

    return shapenet_collate