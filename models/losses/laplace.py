import torch


def laplace_coord(inputs, lap_idx):
    """
    :param inputs: nodes Tensor, size (n_pts, n_features)
    :param lap_idx: laplace index matrix Tensor, size (n_pts, 10)
    for each vertex, the laplace vector shows: [neighbor_index * 8, self, weight]

    :returns
    The laplacian coordinates of input with respect to edges as in lap_idx
    """

    vertex = torch.cat((inputs, torch.zeros(1, 3, dtype=inputs.dtype, device=inputs.device)), 0)

    indices = lap_idx[:, :8]
    weights = lap_idx[:, -1].float()

    weights = torch.reciprocal(weights).reshape((-1, 1)).repeat((1, 3))

    num_pts, num_indices = indices.shape[0], indices.shape[1]
    indices = indices.reshape((-1,))
    vertices = torch.index_select(vertex, 0, indices)
    vertices = vertices.reshape((num_pts, num_indices, 3))

    laplace = torch.sum(vertices, 1)
    laplace = inputs - torch.mul(laplace, weights)

    return laplace


def laplace_regularization(input1, input2, lap_idx, add_move_loss):
    """
    :param input1: vertices tensor before deformation
    :param input2: vertices after the deformation
    :param lap_idx: laplace index matrix tensor
    :param add_move_loss: use move loss
    :return:

    if different than 1 then adds a move loss as in the original TF code
    """

    lap1 = laplace_coord(input1, lap_idx)
    lap2 = laplace_coord(input2, lap_idx)
    laplace_loss = torch.mean(torch.sum(torch.pow(lap1 - lap2, 2), 1)) * 1500
    move_loss = torch.mean(torch.sum(torch.pow(input1 - input2, 2), 1)) * 100
    if add_move_loss:
        return laplace_loss + move_loss
    else:
        return laplace_loss
