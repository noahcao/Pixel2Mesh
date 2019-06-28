import torch.nn as nn

from models.layers.gconv import GConv


class GResBlock(nn.Module):

    def __init__(self, in_dim, hidden_dim, adj_mat):
        super(GResBlock, self).__init__()

        self.conv1 = GConv(in_features=in_dim, out_features=hidden_dim, adj_mat=adj_mat)
        self.conv2 = GConv(in_features=in_dim, out_features=hidden_dim, adj_mat=adj_mat)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)

        return (inputs + x) * 0.5


class GBottleneck(nn.Module):

    def __init__(self, block_num, in_dim, hidden_dim, out_dim, adj_mat):
        super(GBottleneck, self).__init__()

        blocks = [GResBlock(in_dim=hidden_dim, hidden_dim=hidden_dim, adj_mat=adj_mat)]

        for _ in range(block_num - 1):
            blocks.append(GResBlock(in_dim=hidden_dim, hidden_dim=hidden_dim, adj_mat=adj_mat))

        self.blocks = nn.Sequential(*blocks)
        self.conv1 = GConv(in_features=in_dim, out_features=hidden_dim, adj_mat=adj_mat)
        self.conv2 = GConv(in_features=hidden_dim, out_features=out_dim, adj_mat=adj_mat)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x_hidden = self.blocks(x)
        x_out = self.conv2(x_hidden)

        return x_out, x_hidden
