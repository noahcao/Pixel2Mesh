import os

# dataset root
DATASET_ROOT = "datasets/data"
SHAPENET_ROOT = os.path.join(DATASET_ROOT, "shapenet")

# ellipsoid path
ELLIPSOID_PATH = os.path.join(DATASET_ROOT, "ellipsoid/info_ellipsoid.dat")

# pretrained weights path
PRETRAINED_WEIGHTS_PATH = {
    "vgg16": os.path.join(DATASET_ROOT, "pretrained/vgg16-397923af.pth"),
    "resnet50": os.path.join(DATASET_ROOT, "pretrained/resnet50-19c8e357.pth")
}