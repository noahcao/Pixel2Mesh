import torch
import torch.nn as nn

from models.layers.dist_chamfer import ChamferDist
from models.losses.edges import edge_regularization
from models.losses.laplace import laplace_regularization


class P2MLoss(nn.Module):
    def __init__(self, ellipsoid):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='none')
        self.l2_loss = nn.MSELoss(reduction='none')
        self.chamfer_dist = ChamferDist()
        self.ellipsoid = ellipsoid

    def forward(self, pred_pts_list, pred_feats_list, gt_pts):
        """
        :param pred_pts_list: [x1, x1_2, x2, x2_2, x3]
        :param pred_feats_list:
        :param gt_pts:
        :param ellipsoid:
        :return:
        """

        chamfer_loss, edge_loss, lap_loss = 0., 0., 0.
        lap_const = [0.2, 1., 1.]

        for i in range(3):
            dist1, dist2 = self.chamfer_dist(gt_pts, pred_pts_list[i].unsqueeze(0))
            chamfer_loss += torch.mean(dist1) + torch.mean(dist2)
            edge_loss += edge_regularization(pred_pts_list[i], gt_pts, self.ellipsoid.edges[i])
            lap_loss += lap_const[i] * laplace_regularization(pred_feats_list[i], pred_pts_list[i],
                                                              self.ellipsoid.lap_idx[i])

        loss = 100 * chamfer_loss + 0.1 * edge_loss + 0.3 * lap_loss
        return loss, {
            "loss_chamfer": chamfer_loss,
            "loss_edge": edge_loss,
            "loss_laplace": lap_loss
        }
