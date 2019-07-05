import torch
import torch.nn as nn

from models.layers.gbottleneck import GBottleneck
from models.layers.gconv import GConv
from models.layers.gpooling import GUnpooling
from models.layers.gprojection import GProjection
from models.resnet import resnet50


class P2MModel(nn.Module):
    """
    Implement the joint model for Pixel2mesh
    """

    def __init__(self, hidden_dim, coord_dim, ellipsoid):
        super(P2MModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim
        self.init_pts = nn.Parameter(ellipsoid.coord, requires_grad=False)

        self.nn_encoder = resnet50()
        self.features_dim = self.nn_encoder.features_dim + self.coord_dim

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
        batch_size = img.size(0)
        img_feats = self.nn_encoder(img)

        init_pts = self.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)
        # GCN Block 1
        x = self.projection(img_feats, init_pts)
        x1, x_hidden = self.gcns[0](x)

        # before deformation 2
        x1_up = self.unpooling[0](x1)

        # GCN Block 2
        x = self.projection(img_feats, x1)
        x = self.unpooling[0](torch.cat([x, x_hidden], 2))
        # after deformation 2
        x2, x_hidden = self.gcns[1](x)

        # before deformation 3
        x2_up = self.unpooling[1](x2)

        # GCN Block 3
        x = self.projection(img_feats, x2)
        x = self.unpooling[1](torch.cat([x, x_hidden], 2))
        x3, _ = self.gcns[2](x)
        # after deformation 3
        x3 = self.gconv(x3)

        return {
            "pred_coord": [x1, x2, x3],
            "pred_coord_before_deform": [init_pts, x1_up, x2_up],
        }
