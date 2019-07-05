import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class GProjection(nn.Module):
    """Graph Projection layer, which pool 2D features to mesh

    The layer projects a vertex of the mesh to the 2D image and use
    bilinear interpolation to get the corresponding feature.
    """

    def __init__(self):
        super(GProjection, self).__init__()

    def forward(self, img_features, inputs):
        # map to [-1, 1]
        w = (-config.CAMERA_F[0] * (inputs[:, :, 0] / inputs[:, :, 2])) / config.CAMERA_C[0]
        h = (config.CAMERA_F[1] * (inputs[:, :, 1] / inputs[:, :, 2])) / config.CAMERA_C[1]

        # clamp to [-1, 1]
        w = torch.clamp(w, min=-1, max=1)
        h = torch.clamp(h, min=-1, max=1)

        feats = [inputs]
        for i in range(4):
            feats.append(self.project(img_features[i], torch.stack([w, h], dim=-1)))

        output = torch.cat(feats, 2)

        return output

    def project(self, img_feat, sample_points):
        """
        :param img_feat: [batch_size x channel x h x w]
        :param sample_points: [batch_size x num_points x 2], in range [-1, 1]
        :return: [batch_size x num_points x feat_dim]
        """

        output = F.grid_sample(img_feat, sample_points.unsqueeze(1))
        output = torch.transpose(output.squeeze(2), 1, 2)
        return output