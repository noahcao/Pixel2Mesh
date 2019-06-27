import pickle

import numpy as np
import torch

import config


class Ellipsoid(object):

    def __init__(self, file=config.ELLIPSOID_PATH):
        with open(file, "rb") as fp:
            fp_info = pickle.load(fp, encoding='latin1')

        # shape: n_pts * 3
        init_coord = torch.tensor(fp_info[0], dtype=torch.long)

        # edges & faces & lap_idx
        # edge: num_edges * 2
        # faces: num_faces * 4
        # lap_idx: num_pts * 10
        edges, faces, lap_idx = [], [], []

        for i in range(3):
            edges.append(torch.tensor(fp_info[1 + i][1][0], dtype=torch.long))
            faces.append(torch.tensor(fp_info[5][i], dtype=torch.long))
            lap_idx_curr = fp_info[7][i]
            idx = lap_idx_curr.shape[0]
            np.place(lap_idx_curr, lap_idx_curr == -1, idx)
            lap_idx.append(torch.tensor(lap_idx_curr, dtype=torch.long))

        # pool index
        # num_pool_edges * 2
        # pool_01: 462 * 2, pool_12: 1848 * 2
        pool_idx = [torch.tensor(fp_info[4][0], dtype=torch.long), torch.tensor(fp_info[4][1], dtype=torch.long)]

        # supports
        # 0: np.array, 2D, pos
        # 1: np.array, 1D, vals
        # 2: tuple - shape, n * n
        support1, support2, support3 = [], [], []

        for i in range(2):
            support1.append(fp_info[1][i])
            support2.append(fp_info[2][i])
            support3.append(fp_info[3][i])

        self.coord = init_coord
        self.edges = edges
        self.faces = faces
        self.lap_idx = lap_idx
        self.pool_idx = pool_idx
        self.supports = [support1, support2, support3]
