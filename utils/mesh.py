import pickle

import numpy as np
import torch
from scipy.sparse import coo_matrix

import config


def torch_sparse_tensor(indices, value, size):
    coo = coo_matrix((value, (indices[:, 0], indices[:, 1])), shape=size)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.tensor(indices, dtype=torch.long)
    v = torch.tensor(values, dtype=torch.float)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, shape)


class Ellipsoid(object):

    def __init__(self, file=config.ELLIPSOID_PATH):
        with open(file, "rb") as fp:
            fp_info = pickle.load(fp, encoding='latin1')

        # shape: n_pts * 3
        self.coord = torch.tensor(fp_info[0])

        # edges & faces & lap_idx
        # edge: num_edges * 2
        # faces: num_faces * 4
        # laplace_idx: num_pts * 10
        self.edges, self.faces, self.laplace_idx = [], [], []

        for i in range(3):
            self.edges.append(torch.tensor(fp_info[1 + i][1][0], dtype=torch.long))
            self.faces.append(torch.tensor(fp_info[5][i], dtype=torch.long))
            lap_idx_curr = fp_info[7][i]
            idx = lap_idx_curr.shape[0]
            np.place(lap_idx_curr, lap_idx_curr == -1, idx)
            self.laplace_idx.append(torch.tensor(lap_idx_curr, dtype=torch.long))

        # unpool index
        # num_pool_edges * 2
        # pool_01: 462 * 2, pool_12: 1848 * 2
        self.unpool_idx = [torch.tensor(fp_info[4][i], dtype=torch.long) for i in range(2)]

        # loops and adjacent edges
        self.adj_mat = []
        for i in range(1, 4):
            # 0: np.array, 2D, pos
            # 1: np.array, 1D, vals
            # 2: tuple - shape, n * n
            adj_loops = torch_sparse_tensor(*fp_info[i][0])
            adj_mat = torch_sparse_tensor(*fp_info[i][1])
            adj_mat += adj_loops
            self.adj_mat.append(adj_mat)
