import torch
import torch.nn as nn
import numpy as np


class GUnpooling(nn.Module):
    """Graph Pooling layer, aims to add additional vertices to the graph.
    The middle point of each edges are added, and its feature is simply
    the average of the two edge vertices.
    Three middle points are connected in each triangle.
    """

    def __init__(self, pool_idx):
        super(GUnpooling, self).__init__()
        self.pool_idx = pool_idx
        # save dim info
        self.in_num = torch.max(pool_idx).item()
        self.out_num = self.in_num + len(pool_idx)

    def forward(self, input):
        new_features = input[self.pool_idx].clone()
        new_vertices = 0.5 * new_features.sum(1)
        output = torch.cat([input, new_vertices], 0)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_num) + ' -> ' \
               + str(self.out_num) + ')'