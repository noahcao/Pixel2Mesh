import torch


def edge_regularization(pred, gt_pts, edges):
    idx1 = edges[:, 0]
    idx2 = torch.tensor(edges[:, 1]).long()

    nod1 = torch.index_select(pred, 0, idx1)
    nod2 = torch.index_select(pred, 0, idx2)
    edge = nod1 - nod2

    # edge length loss
    edge_length = torch.sum(torch.pow(edge, 2), 1)
    edge_loss = torch.mean(edge_length) * 300

    return edge_loss
