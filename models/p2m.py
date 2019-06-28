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

    def __init__(self, features_dim, hidden_dim, coord_dim, ellipsoid):
        super(P2MModel, self).__init__()
        self.img_size = 224

        self.features_dim = features_dim
        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim
        self.init_pts = nn.Parameter(ellipsoid.coord, requires_grad=False)

        self.nn_encoder = self.build_encoder()
        self.nn_decoder = self.build_decoder()

        self.gcns = nn.ModuleList([
            GBottleneck(6, self.features_dim, self.hidden_dim, self.coord_dim,
                        ellipsoid.adj_mat[0]),
            GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.coord_dim,
                        ellipsoid.adj_mat[1]),
            GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.hidden_dim,
                        ellipsoid.adj_mat[2])
        ])

        self.unpooling = nn.ModuleList([
            GUnpooling(ellipsoid.unpool_idx[0]),
            GUnpooling(ellipsoid.unpool_idx[1])
        ])

        self.projection = GProjection()

        self.gconv = GConv(in_features=self.hidden_dim, out_features=self.coord_dim,
                           adj_mat=ellipsoid.adj_mat[2])

    def forward(self, img):
        img_feats = self.nn_encoder(img)

        # GCN Block 1
        x = self.projection(img_feats, self.init_pts.data)
        x1, x_hidden = self.gcns[0](x)

        # before deformation 2
        x1_up = self.unpooling[0](x1)

        # GCN Block 2
        x = self.projection(img_feats, x1)
        x = self.unpooling[0](torch.cat([x, x_hidden], 1))
        # after deformation 2
        x2, x_hidden = self.gcns[1](x)

        # before deformation 3
        x2_up = self.unpooling[1](x2)

        # GCN Block 3
        x = self.projection(img_feats, x2)
        x = self.unpooling[1](torch.cat([x, x_hidden], 1))
        x3, _ = self.gcns[2](x)
        # after deformation 3
        x3 = self.gconv(x3)

        img_recons = self.nn_decoder(img_feats)

        return {
            "pred_coord": [x1, x2, x3],
            "pred_coord_before_deform": [self.init_pts.data, x1_up, x2_up],
            "img_recons": img_recons
        }

    def build_encoder(self):
        # VGG16 at first, then try resnet
        # Can load params from model zoo
        net = VGG16P2M(n_classes_input=3)
        return net

    def build_decoder(self):
        net = VGG16Decoder()
        return net
