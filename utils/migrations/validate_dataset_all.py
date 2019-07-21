import os
import sys

import requests
from tqdm import tqdm


def go(file_path, subset):
    shapenet_root = "datasets/data/shapenet"
    with open(file_path, "r") as f, open(os.path.join(shapenet_root, "meta", subset + "_all.txt"), "w") as g:
        for line in tqdm(f.readlines()):
            _, _, label, filename, _, index = line.strip().split("/")
            converted = label + "_" + filename + "_" + index
            file_path = os.path.join(shapenet_root, "data", label + "/" + filename + "_" + index)
            if not os.path.exists(file_path):
                print("fail! " + file_path)
                continue
            print(converted, file=g)


go(sys.argv[1], "train")
go(sys.argv[2], "test")