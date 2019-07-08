import cv2
import neural_renderer as nr
import numpy as np
import torch

import config


def _process_render_result(img, height, width):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.ndim == 2:
        # assuming single channel image
        img = np.expand_dims(img, axis=0)
    if img.shape[-1] == 3:
        # assuming [height, width, rgb]
        img = np.moveaxis(img, -1, 0)
    # return 3 * width * height or width * height, in range [0, 1]
    return np.clip(img[:height, :width], 0, 1)


def _mix_render_result_with_image(rgb, alpha, image):
    alpha = np.expand_dims(alpha, 0)
    return alpha * rgb + (1 - alpha) * image


class MeshRenderer(object):

    def __init__(self):
        self.colors = {'pink': np.array([.9, .7, .7]),
                       'light_blue': np.array([0.65098039, 0.74117647, 0.85882353])
                       }
        self.renderer = nr.Renderer(camera_mode='projection',
                                    light_intensity_directional=.8,
                                    light_intensity_ambient=.3,
                                    background_color=[1., 1., 1.],
                                    light_direction=[0., 0., -1.])

    def _render_mesh(self, vertices: np.ndarray, faces: np.ndarray, width, height,
                     camera_k, camera_dist_coeffs, rvec, tvec, color=None):
        # render a square image, then crop
        img_size = max(height, width)

        # This is not thread safe!
        self.renderer.image_size = img_size

        vertices = torch.tensor(vertices, dtype=torch.float32)
        faces = torch.tensor(faces, dtype=torch.int32)

        color = self.colors[color or 'light_blue']
        texture_size = 2
        textures = torch.tensor(color, dtype=torch.float32) \
            .repeat(faces.size(0), texture_size, texture_size, texture_size, 1)

        camera_k = torch.tensor(camera_k, dtype=torch.float32)
        rotmat = torch.tensor(cv2.Rodrigues(rvec)[0], dtype=torch.float32)
        tvec = torch.tensor(tvec, dtype=torch.float32)
        camera_dist_coeffs = torch.tensor(camera_dist_coeffs, dtype=torch.float32)

        rgb, _, alpha = self.renderer.render(vertices.unsqueeze(0).cuda(),
                                             faces.unsqueeze(0).cuda(),
                                             textures.unsqueeze(0).cuda(),
                                             K=camera_k.unsqueeze(0).cuda(),
                                             R=rotmat.unsqueeze(0).cuda(),
                                             t=tvec.unsqueeze(0).cuda(),
                                             dist_coeffs=camera_dist_coeffs.unsqueeze(0).cuda(),
                                             orig_size=img_size)
        # use the extra dimension of alpha for broadcasting
        alpha = _process_render_result(alpha[0], height, width)
        rgb = _process_render_result(rgb[0], height, width)

        return rgb, alpha

    def _render_pointcloud(self, vertices: np.ndarray, width, height,
                           camera_k, camera_dist_coeffs, rvec, tvec, color=None):
        color = self.colors[color or 'pink']

        # return pointcloud
        vertices_2d = cv2.projectPoints(np.expand_dims(vertices, -1),
                                        rvec, tvec, camera_k, camera_dist_coeffs)[0]
        vertices_2d = np.reshape(vertices_2d, (-1, 2))
        alpha = np.zeros((height, width, 3), np.float)
        whiteboard = np.ones((3, height, width), np.float)
        if np.isnan(vertices_2d).any():
            return whiteboard, alpha
        for x, y in vertices_2d:
            cv2.circle(alpha, (x, y), radius=1, color=(1., 1., 1.), thickness=-1)
        rgb = _process_render_result(alpha * color[None, None, :], height, width)
        alpha = _process_render_result(alpha[:, :, 0], height, width)
        rgb = _mix_render_result_with_image(rgb, alpha[0], whiteboard)
        return rgb, alpha

    def visualize_reconstruction(self, gt_coord, coord, faces, image):
        camera_k = np.array([[config.CAMERA_F[0], 0, config.CAMERA_C[0]],
                             [0, config.CAMERA_F[1], config.CAMERA_C[1]],
                             [0, 0, 1]])
        # inverse y and z, equivalent to inverse x, but gives positive z
        rvec = np.array([np.pi, 0., 0.], dtype=np.float32)
        tvec = np.zeros(3, dtype=np.float32)
        dist_coeffs = np.zeros(5, dtype=np.float32)
        gt_pc, _ = self._render_pointcloud(gt_coord, image.shape[2], image.shape[1],
                                           camera_k, dist_coeffs, rvec, tvec)
        pred_pc, _ = self._render_pointcloud(coord, image.shape[2], image.shape[1],
                                             camera_k, dist_coeffs, rvec, tvec)
        mesh, _ = self._render_mesh(coord, faces, image.shape[2], image.shape[1],
                                    camera_k, dist_coeffs, rvec, tvec)
        return np.concatenate((image, gt_pc, pred_pc, mesh), 2)

    def p2m_batch_visualize(self, batch_input, batch_output, faces, atmost=3):
        """
        Every thing is tensor for now, needs to move to cpu and convert to numpy
        """
        batch_size = min(batch_input["images"].size(0), atmost)
        images_stack = []
        for i in range(batch_size):
            image = batch_input["images"][i].cpu().numpy()
            gt_points = batch_input["points"][i].cpu().numpy()
            for j in range(3):
                for k in (["pred_coord_before_deform", "pred_coord"] if j == 0 else ["pred_coord"]):
                    coord = batch_output[k][j][i].cpu().numpy()
                    images_stack.append(self.visualize_reconstruction(gt_points, coord, faces[j].cpu().numpy(), image))
        return torch.from_numpy(np.concatenate(images_stack, 1))
