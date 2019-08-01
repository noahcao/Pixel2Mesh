import json
import os
import random
import shutil

with open("datasets/data/shapenet/meta/shapenet.json") as fp:
    labels_map = json.load(fp)

with open("datasets/data/shapenet/meta/test_tf.txt") as fp:
    lines = [line.strip() for line in fp.readlines()]

for entry in labels_map.values():
    file_list = list(filter(lambda x: (entry["id"] + "/") in x, lines))
    chosen = random.choice(file_list)
    file_location = os.path.join("datasets/data/shapenet/data_tf",
                                 chosen[len("Data/ShapeNetP2M/"):-4] + ".png")
    shutil.copyfile(file_location, "datasets/examples/%s.png" % entry["name"].split(",")[0])
