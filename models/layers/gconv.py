import math

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix


def torch_sparse_tensor(indice, value, size, use_cuda):
    coo = coo_matrix((value, (indice[:, 0], indice[:, 1])), shape = size)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    if use_cuda:
        return torch.sparse.FloatTensor(i, v, shape).cuda()
    else:
        return torch.sparse.FloatTensor(i, v, shape)

def dot(x, y, sparse = False):
    """Wrapper for torch.matmul (sparse vs dense)."""
    if sparse:
        res = x.mm(y)
    else:
        res = torch.matmul(x, y)
    return res


class GConv(nn.Module):
    """Simple GCN layer

    Similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, adjs, bias=True, use_cuda=True):
        super(GConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        adj0 = torch_sparse_tensor(*adjs[0], use_cuda)
        adj1 = torch_sparse_tensor(*adjs[1], use_cuda)
        self.adjs = [adj0, adj1]

        self.weight = nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features, ), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        # output = torch.spmm(adj, support)
        output1 = dot(self.adjs[0], support, True)
        output2 = dot(self.adjs[1], support, True)
        output = output1 + output2
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
