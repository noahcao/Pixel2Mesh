import torch


def laplace_coord(input, lap_idx):
    """
    :param input: nodes Tensor, size (n_pts, n_features)
    :param lap_idx: laplace index matrix Tensor, size (n_pts, 10)
    for each vertex, the laplace vector shows: [neighbor_index * 8, self, weight]

    :returns
    The laplacian coordinates of input with respect to edges as in lap_idx
    """

    vertex = torch.cat((input, torch.zeros(1, 3)), 0)

    indices = lap_idx[:, :8]
    weights = lap_idx[:, -1]

    weights = torch.reciprocal(weights).reshape((-1, 1)).repeat((1, 3))

    num_pts, num_indices = indices.shape[0], indices.shape[1]
    indices = indices.reshape((-1,))
    vertices = torch.index_select(vertex, 0, indices)
    vertices = vertices.reshape((num_pts, num_indices, 3))

    laplace = torch.sum(vertices, 1)
    laplace = input - torch.mul(laplace, weights)

    return laplace


def laplace_regularization(input1, input2, lap_idx):
    """
    :param input1: vertices tensor before deformation
    :param input2: vertices after the deformation
    :param lap_idx: laplace index matrix tensor
    :return:

    if different than 1 then adds a move loss as in the original TF code
    """

    lap1 = laplace_coord(input1, lap_idx)
    lap2 = laplace_coord(input2, lap_idx)
    laplace_loss = torch.mean(torch.sum(torch.pow(lap1 - lap2, 2), 1)) * 1500
    move_loss = torch.mean(torch.sum(torch.pow(input1 - input2, 2), 1)) * 100

    if block_id == 0:
        return laplace_loss
    else:
        return laplace_loss + move_loss