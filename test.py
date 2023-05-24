import torch
import torch.nn as nn

from models.layers.chamfer_wrapper import ChamferDist
from models.losses.p2m import emd_loss

def test():
    torch.manual_seed(42)
    dense = nn.Linear(6, 3)
    dense.cuda()
    optimizer = torch.optim.Adam(dense.parameters(), 1e-3)
    a = torch.rand(4, 5, 6).cuda()
    b = torch.rand(4, 8, 3).cuda()
    c = torch.rand(4, 5, 6).cuda()
    for i in range(30000):
        a_out = dense(a)

        # Compute EMD_loss instead of Chamfer loss
        loss = emd_loss(a_out, b)

        c_out = dense(a)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)


test()