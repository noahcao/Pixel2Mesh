import pickle
import torch
import numpy as np


with open("checkpoints/debug/migration/p2m-tensorflow.pkl", "rb") as f:
    official = pickle.load(f)
for k, v in official.items():
    print(k, v.shape)

with open("checkpoints/debug/host_template_256/000001_000001.pt", "rb") as f:
    host = torch.load(f)
for k, v in host["model"].items():
    print(k, v.shape)

with open("utils/migrations/official_config_pytorch_256.txt", "r") as f:
    pt_names = [line.split()[0] for line in f.readlines()]
with open("utils/migrations/official_config_tensorflow_256.txt", "r") as f:
    tf_names = [line.split()[0] for line in f.readlines()]
for pt, tf in zip(pt_names, tf_names):
    if host["model"][pt].shape != official[tf].shape:
        data = np.transpose(official[tf], (3, 2, 0, 1))
    else:
        data = official[tf]
    print(pt, tf, host["model"][pt].data.shape, data.shape)
    host["model"][pt].data = torch.from_numpy(data)

torch.save(host, "checkpoints/debug/migration/network_official.pt")