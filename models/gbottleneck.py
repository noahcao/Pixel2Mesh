import torch.nn as nn

from models.layers.gconv import GConv


class GResBlock(nn.Module):

    def __init__(self, in_dim, hidden_dim, adjs, use_cuda):
        super(GResBlock, self).__init__()

        self.conv1 = GConv(in_features=in_dim, out_features=hidden_dim, adjs=adjs, use_cuda=use_cuda)
        self.conv2 = GConv(in_features=in_dim, out_features=hidden_dim, adjs=adjs, use_cuda=use_cuda)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)

        return (input + x) * 0.5


class GBottleneck(nn.Module):

    def __init__(self, block_num, in_dim, hidden_dim, out_dim, adjs, use_cuda):
        super(GBottleneck, self).__init__()

        blocks = [GResBlock(in_dim=hidden_dim, hidden_dim=hidden_dim, adjs=adjs, use_cuda=use_cuda)]

        for _ in range(block_num - 1):
            blocks.append(GResBlock(in_dim=hidden_dim, hidden_dim=hidden_dim, adjs=adjs, use_cuda=use_cuda))

        self.blocks = nn.Sequential(*blocks)
        self.conv1 = GConv(in_features=in_dim, out_features=hidden_dim, adjs=adjs, use_cuda=use_cuda)
        self.conv2 = GConv(in_features=hidden_dim, out_features=out_dim, adjs=adjs, use_cuda=use_cuda)

    def forward(self, input):
        x = self.conv1(input)
        x_cat = self.blocks(x)
        x_out = self.conv2(x_cat)

        return x_out, x_cat
