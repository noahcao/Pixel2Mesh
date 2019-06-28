import torch
import torch.nn as nn

from models.layers.dist_chamfer import ChamferDist
from models.losses.laplace import laplace_regularization


class P2MLoss(nn.Module):
    def __init__(self, ellipsoid):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='none')
        self.l2_loss = nn.MSELoss(reduction='none')
        self.chamfer_dist = ChamferDist()
        self.laplace_idx = nn.ParameterList([
            nn.Parameter(idx, requires_grad=False) for idx in ellipsoid.laplace_idx])
        self.edges = nn.ParameterList([
            nn.Parameter(edges, requires_grad=False) for edges in ellipsoid.edges])

    @staticmethod
    def edge_regularization(pred, edges):
        edge = pred[edges[:, 0]] - pred[edges[:, 1]]

        edge_length = torch.sum(torch.pow(edge, 2), 1)
        return torch.mean(edge_length) * 300

    def forward(self, outputs, targets):
        """
        :param outputs: outputs from P2MModel
        :param targets: targets from input
        :return: loss, loss_summary (dict)
        """

        chamfer_loss, edge_loss, lap_loss = 0., 0., 0.
        lap_const = [0.2, 1., 1.]

        gt_coord = targets["points"]
        pred_coord, pred_coord_before_deform = outputs["pred_coord"], outputs["pred_coord_before_deform"]

        for i in range(3):
            dist1, dist2 = self.chamfer_dist(gt_coord, pred_coord_before_deform[i].unsqueeze(0))
            chamfer_loss += torch.mean(dist1) + torch.mean(dist2)
            edge_loss += self.edge_regularization(pred_coord[i], self.edges[i])
            lap_loss += lap_const[i] * laplace_regularization(pred_coord_before_deform[i], pred_coord[i],
                                                              self.laplace_idx[i], i > 0)

        loss = 100 * chamfer_loss + 0.1 * edge_loss + 0.3 * lap_loss
        return loss, {
            "loss": loss,
            "loss_chamfer": chamfer_loss,
            "loss_edge": edge_loss,
            "loss_laplace": lap_loss
        }
