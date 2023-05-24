# -*- coding: utf-8 -*-
import numpy as np
import torch
import cv2


def normalize_3D(points: torch.tensor):
    centroid = torch.mean(points)
    points -= centroid
    distances = torch.linalg.norm(points, dim=1)
    maximum_extent = torch.max(distances)
    points /= maximum_extent
    return points


def calculate_focal(points: torch.tensor, height: int, width: int):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    focal = (0.5 * torch.min(z) * min(height, width)) / max(torch.max(x), torch.max(y))
    return focal.item()


def project_matrix(focal: int, height: int, width: int):
    pm = torch.tensor([
        [focal, 0, height / 2],
        [0, focal, width / 2],
        [0, 0, 1]
    ], dtype=torch.float32)
    return pm


def project_3D_to_2D(points: torch.tensor, rvec: torch.tensor, camera_matrix: np.array):
    points = points.detach().cpu().numpy()
    camera_matrix = camera_matrix.detach().cpu().numpy()

    dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)

    # rvec = np.array([0, 0, 0], dtype=np.float32)
    tvec = np.array([0, 0, 0], dtype=np.float32)

    image_points, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
    return torch.from_numpy(image_points)


def calculate_loss(gt_points: torch.tensor, pred_points: torch.tensor, height: int, width: int, l2_loss):
    gt_points = torch.round(gt_points)
    gt_points = gt_points.reshape(-1, 2)

    pred_points = torch.round(pred_points)
    pred_points = pred_points.reshape(-1, 2)

    gt_pixels = torch.zeros((height, width))
    pred_pixels = torch.zeros((height, width))

    x_min = -192 // 2
    x_max = 192 // 2
    y_min = -256 // 2
    y_max = 256 // 2

    # Find gt pixels
    mask = (gt_points[:, 0] >= x_min) & (gt_points[:, 0] <= x_max) & (gt_points[:, 1] >= y_min) & (
                gt_points[:, 1] <= y_max)
    x_indices = (gt_points[mask, 0] + x_max - 1).to(torch.int)
    y_indices = (gt_points[mask, 1] + y_max - 1).to(torch.int)
    gt_pixels[x_indices, y_indices] = 1

    # Find pred pixels
    mask = (pred_points[:, 0] >= x_min) & (pred_points[:, 0] <= x_max) & (pred_points[:, 1] >= y_min) & (
                pred_points[:, 1] <= y_max)
    x_indices = (pred_points[mask, 0] + x_max - 1).to(torch.int)
    y_indices = (pred_points[mask, 1] + y_max - 1).to(torch.int)
    pred_pixels[x_indices, y_indices] = 1

    loss = l2_loss(gt_pixels, pred_pixels)
    return loss


def geometric_loss(gt_points: torch.tensor, pred_points: torch.tensor, height: int, width: int):
    total_loss = 0
    l2_loss = torch.nn.MSELoss()
    for views in range(3):
        rvec = np.random.randint(low=0, high=2 * 3.14, size=(3,)).astype(np.float64)
        #print(f"rotation vector is: ", rvec)

        gt_normal = torch_normalize(gt_points)
        gt_focal = torch_focal(gt_normal, height, width)
        gt_pm = torch_matrix(gt_focal, height, width)
        gt_projection = torch_project_3D_to_2D(gt_normal, rvec, gt_pm)

        pred_normal = torch_normalize(pred_points)
        pred_focal = torch_focal(pred_normal, height, width)
        pred_pm = torch_matrix(pred_focal, height, width)
        pred_projection = torch_project_3D_to_2D(pred_normal, rvec, pred_pm)

        loss = torch_calculate_loss(gt_projection, pred_projection, height, width, l2_loss)
        total_loss += loss

    return total_loss.item()
