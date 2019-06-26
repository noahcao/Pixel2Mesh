import torch
import torch.nn as nn

from models.gbottleneck import GBottleneck
from models.layers.gconv import GConv
from models.layers.gpooling import GUnpooling
from models.layers.gprojection import GProjection
from models.vgg16 import VGG16P2M, VGG16Decoder


class P2MModel(nn.Module):
    """
    Implement the joint model for Pixel2mesh
    """

    def __init__(self, features_dim, hidden_dim, coord_dim, pool_idx, supports, use_cuda):
        super(P2MModel, self).__init__()
        self.img_size = 224

        self.features_dim = features_dim
        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim
        self.pool_idx = pool_idx
        self.supports = supports
        self.use_cuda = use_cuda

        self.build()

    def build(self):
        self.nn_encoder = self.build_encoder()
        self.nn_decoder = self.build_decoder()

        self.GCN_0 = GBottleneck(6, self.features_dim, self.hidden_dim, self.coord_dim, self.supports[0], self.use_cuda)
        self.GCN_1 = GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.coord_dim,
                                 self.supports[1], self.use_cuda)
        self.GCN_2 = GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.hidden_dim,
                                 self.supports[2], self.use_cuda)

        self.GPL_1 = GUnpooling(self.pool_idx[0])
        self.GPL_2 = GUnpooling(self.pool_idx[1])

        self.GPR_0 = GProjection()
        self.GPR_1 = GProjection()
        self.GPR_2 = GProjection()

        self.GConv = GConv(in_features=self.hidden_dim, out_features=self.coord_dim, adjs=self.supports[2],
                                      use_cuda=self.use_cuda)

        self.GPL_12 = GUnpooling(self.pool_idx[0])
        self.GPL_22 = GUnpooling(self.pool_idx[1])

    def forward(self, img, input):
        img_feats = self.nn_encoder(img)

        # GCN Block 1
        x = self.GPR_0(img_feats, input)
        x1, x_cat = self.GCN_0(x)
        x1_2 = self.GPL_12(x1)

        # GCN Block 2
        x = self.GPR_1(img_feats, x1)
        x = torch.cat([x, x_cat], 1)
        x = self.GPL_1(x)
        x2, x_cat = self.GCN_1(x)
        x2_2 = self.GPL_22(x2)

        # GCN Block 3
        x = self.GPR_2(img_feats, x2)
        x = torch.cat([x, x_cat], 1)
        x = self.GPL_2(x)
        x, _ = self.GCN_2(x)

        x3 = self.GConv(x)

        new_img = self.nn_decoder(img_feats)

        return [x1, x2, x3], [input, x1_2, x2_2], new_img

    def build_encoder(self):
        # VGG16 at first, then try resnet
        # Can load params from model zoo
        net = VGG16P2M(n_classes_input=3)
        return net

    def build_decoder(self):
        net = VGG16Decoder()
        return net
