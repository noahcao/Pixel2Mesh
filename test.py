import torch
import torch.nn as nn

from models.layers.chamfer_wrapper import ChamferDist


def test():
    torch.manual_seed(42)
    chamfer = ChamferDist()
    dense = nn.Linear(6, 3)
    dense.cuda()
    optimizer = torch.optim.Adam(dense.parameters(), 1e-3)
    a = torch.rand(4, 5, 6).cuda()
    b = torch.rand(4, 8, 3).cuda()
    c = torch.rand(4, 5, 6).cuda()
    for i in range(30000):
        a_out = dense(a)
        d1, d2, i1, i2 = chamfer(a_out, b)
        loss = d1.mean() + d2.mean()

        c_out = dense(a)
        d1, d2, i1, i2 = chamfer(c_out, b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)


test()