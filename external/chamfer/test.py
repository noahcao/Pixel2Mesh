import sys
import os
for file in os.listdir("build"):
    if file.startswith("lib"):
        sys.path.insert(0, os.path.join("build", file))

# torch must be imported before we import chamfer
import torch
import chamfer

batch_size = 8
n, m = 30, 20

xyz1 = torch.rand((batch_size, n, 3)).cuda()
xyz2 = torch.rand((batch_size, m, 3)).cuda()

dist1 = torch.zeros(batch_size, n).cuda()
dist2 = torch.zeros(batch_size, m).cuda()

idx1 = torch.zeros((batch_size, n), dtype=torch.int).cuda()
idx2 = torch.zeros((batch_size, m), dtype=torch.int).cuda()

chamfer.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
print(dist1)
print(dist2)
print(idx1)
print(idx2)