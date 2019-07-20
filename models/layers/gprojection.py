import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Threshold


class GProjectionCompat(nn.Module):
    """
    Supports only batch_size = 1.
    We use this for debugging.
    Please don't use this!
    """

    def __init__(self, mesh_pos, camera_f, camera_c, bound=0):
        self.mesh_pos, self.camera_f, self.camera_c = mesh_pos, camera_f, camera_c
        super(GProjectionCompat, self).__init__()

    def forward(self, img_features, input):
        self.img_feats = img_features
        input = input[0]

        h = self.camera_f[1] * torch.div(input[:, 1], input[:, 2]) + self.camera_c[1]
        w = self.camera_f[0] * torch.div(input[:, 0], -input[:, 2]) + self.camera_c[0]

        h = torch.clamp(h, min=0, max=223)
        w = torch.clamp(w, min=0, max=223)

        img_sizes = [56, 28, 14, 7]
        out_dims = [64, 128, 256, 512]
        feats = [input]

        for i in range(4):
            out = self.project(i, h, w, img_sizes[i], out_dims[i])
            feats.append(out)

        output = torch.cat(feats, 1)

        return output.unsqueeze(0)

    def project(self, index, h, w, img_size, out_dim):
        img_feat = self.img_feats[index][0]
        x = h / (224. / img_size)
        y = w / (224. / img_size)

        x1, x2 = torch.floor(x).long(), torch.ceil(x).long()
        y1, y2 = torch.floor(y).long(), torch.ceil(y).long()

        x2 = torch.clamp(x2, max=img_size - 1)
        y2 = torch.clamp(y2, max=img_size - 1)

        Q11 = img_feat[:, x1, y1].clone()
        Q12 = img_feat[:, x1, y2].clone()
        Q21 = img_feat[:, x2, y1].clone()
        Q22 = img_feat[:, x2, y2].clone()

        weights = torch.mul(x2.float() - x, y2.float() - y)
        Q11 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q11, 0, 1))

        weights = torch.mul(x2.float() - x, y - y1.float())
        Q12 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q12, 0, 1))

        weights = torch.mul(x - x1.float(), y2.float() - y)
        Q21 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q21, 0, 1))

        weights = torch.mul(x - x1.float(), y - y1.float())
        Q22 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q22, 0, 1))

        output = Q11 + Q21 + Q12 + Q22

        return output


class GProjection(nn.Module):
    """
    Graph Projection layer, which pool 2D features to mesh

    The layer projects a vertex of the mesh to the 2D image and use
    bi-linear interpolation to get the corresponding feature.
    """

    def __init__(self, mesh_pos, camera_f, camera_c, bound=0):
        super(GProjection, self).__init__()
        self.mesh_pos, self.camera_f, self.camera_c = mesh_pos, camera_f, camera_c
        self.threshold = None
        self.bound = 0
        if self.bound != 0:
            self.threshold = Threshold(bound, bound)

    def bound_val(self, x):
        """
        given x, return min(threshold, x), in case threshold is not None
        """
        if self.bound < 0:
            return -self.threshold(-x)
        elif self.bound > 0:
            return self.threshold(x)
        return x

    def forward(self, img_features, inputs):
        # map to [-1, 1]
        # not sure why they render to negative x
        positions = inputs + torch.tensor(self.mesh_pos, device=inputs.device, dtype=torch.float)
        w = (-self.camera_f[0] * (positions[:, :, 0] / self.bound_val(positions[:, :, 2]))) / self.camera_c[0]
        h = (self.camera_f[1] * (positions[:, :, 1] / self.bound_val(positions[:, :, 2]))) / self.camera_c[1]

        # clamp to [-1, 1]
        w = torch.clamp(w, min=-1, max=1)
        h = torch.clamp(h, min=-1, max=1)

        feats = [inputs]
        for img_feature in img_features:
            feats.append(self.project(img_feature, torch.stack([w, h], dim=-1)))

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
