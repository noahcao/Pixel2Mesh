import torch
import torch.nn as nn

from models.layers.dist_chamfer import ChamferDist


class P2MLoss(nn.Module):
    def __init__(self, ellipsoid):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='none')
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.chamfer_dist = ChamferDist()
        self.laplace_idx = nn.ParameterList([
            nn.Parameter(idx, requires_grad=False) for idx in ellipsoid.laplace_idx])
        self.edges = nn.ParameterList([
            nn.Parameter(edges, requires_grad=False) for edges in ellipsoid.edges])

    def edge_regularization(self, pred, edges):
        return self.l2_loss(pred[:, edges[:, 0]], pred[:, edges[:, 1]]) * 300

    @staticmethod
    def laplace_coord(inputs, lap_idx):
        """
        :param inputs: nodes Tensor, size (n_pts, n_features = 3)
        :param lap_idx: laplace index matrix Tensor, size (n_pts, 10)
        for each vertex, the laplace vector shows: [neighbor_index * 8, self_index, neighbor_count]

        :returns
        The laplacian coordinates of input with respect to edges as in lap_idx
        """

        indices = lap_idx[:, :-2]
        invalid_mask = indices < 0
        all_valid_indices = indices.clone()
        all_valid_indices[invalid_mask] = 0  # do this to avoid negative indices

        vertices = inputs[:, all_valid_indices]
        vertices[:, invalid_mask] = 0
        neighbor_sum = torch.sum(vertices, 2)
        neighbor_count = lap_idx[:, -1].float()
        laplace = inputs - neighbor_sum / neighbor_count[None, :, None]

        return laplace

    def laplace_regularization(self, input1, input2, block_idx):
        """
        :param input1: vertices tensor before deformation
        :param input2: vertices after the deformation
        :param block_idx: idx to select laplace index matrix tensor
        :return:

        if different than 1 then adds a move loss as in the original TF code
        """

        lap1 = self.laplace_coord(input1, self.laplace_idx[block_idx])
        lap2 = self.laplace_coord(input2, self.laplace_idx[block_idx])
        laplace_loss = self.l2_loss(lap1, lap2) * 1500
        move_loss = self.l2_loss(input1, input2) * 100
        if block_idx > 0:
            return laplace_loss + move_loss
        else:
            return laplace_loss

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
            dist1, dist2 = self.chamfer_dist(gt_coord, pred_coord[i])
            chamfer_loss += torch.mean(dist1) + torch.mean(dist2)
            edge_loss += self.edge_regularization(pred_coord[i], self.edges[i])
            lap_loss += lap_const[i] * self.laplace_regularization(pred_coord_before_deform[i],
                                                                   pred_coord[i], i)

        loss = 100 * chamfer_loss + 0.1 * edge_loss + 0.3 * lap_loss
        return loss, {
            "loss": loss,
            "loss_chamfer": chamfer_loss,
            "loss_edge": edge_loss,
            "loss_laplace": lap_loss
        }
