import os

import tensorflow as tf

import numpy as np
import pickle
from skimage import io, transform


def create_dataset(filename):
    with open(filename, "r") as f:
        file_list = [os.path.join("data/ShapeNetP2M", s.strip()) for s in f.readlines()]
    return tf.data.Dataset.from_tensor_slices(file_list)


def shapenet_p2m_process(filename):
    filename = filename.decode()
    with open(filename, 'rb') as pkl_file:
        label = pickle.load(pkl_file, encoding="latin1")
    img_path = filename.replace('.dat', '.png')
    img = io.imread(img_path)
    img[np.where(img[:, :, 3] == 0)] = 255
    img = transform.resize(img, (224, 224))
    img = img[:, :, :3].astype('float32')
    return img, label, os.path.basename(filename)
