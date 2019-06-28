import torch
import torch.nn as nn


class GProjection(nn.Module):
    """Graph Projection layer, which pool 2D features to mesh

    The layer projects a vertex of the mesh to the 2D image and use
    bilinear interpolation to get the corresponding feature.
    """

    def __init__(self):
        super(GProjection, self).__init__()

    def forward(self, img_features, input):
        h = 248 * torch.div(input[:, 1], input[:, 2]) + 111.5
        w = -248 * torch.div(input[:, 0], input[:, 2]) + 111.5

        h = torch.clamp(h, min=0, max=223)
        w = torch.clamp(w, min=0, max=223)

        img_sizes = [56, 28, 14, 7]
        out_dims = [64, 128, 256, 512]
        feats = [input]

        for i in range(4):
            out = self.project(i, h, w, img_sizes[i], img_features)
            feats.append(out)

        output = torch.cat(feats, 1)

        return output

    def project(self, index, h, w, img_size, img_features):
        img_feat = img_features[index]
        x = h.float() / (224. / img_size)
        y = w.float() / (224. / img_size)

        x1, x2 = torch.floor(x).long(), torch.ceil(x).long()
        y1, y2 = torch.floor(y).long(), torch.ceil(y).long()

        x2 = torch.clamp(x2, max=img_size - 1)
        y2 = torch.clamp(y2, max=img_size - 1)

        Q11 = img_feat[:, x1, y1].clone()
        Q12 = img_feat[:, x1, y2].clone()
        Q21 = img_feat[:, x2, y1].clone()
        Q22 = img_feat[:, x2, y2].clone()

        x, y = x.long(), y.long()

        weights = torch.mul(x2 - x, y2 - y)
        Q11 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q11, 0, 1))

        weights = torch.mul(x2 - x, y - y1)
        Q12 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q12, 0, 1))

        weights = torch.mul(x - x1, y2 - y)
        Q21 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q21, 0, 1))

        weights = torch.mul(x - x1, y - y1)
        Q22 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q22, 0, 1))

        output = Q11 + Q21 + Q12 + Q22

        return output