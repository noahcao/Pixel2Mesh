import numpy as np
import pyrender
import torch
import trimesh
from pyrender import RenderFlags

import config


# init a global renderer to bypass the problem with multiple renderers
global_renderer = pyrender.OffscreenRenderer(config.CAMERA_RES[0], config.CAMERA_RES[1])


def render_pyrender_mesh(mesh, point_size=1.):
    scene = pyrender.Scene(ambient_light=[.03, .03, .03],
                           bg_color=[0., 0., 0., 0.])
    scene.add(mesh)
    camera = pyrender.IntrinsicsCamera(fx=config.CAMERA_F[0], fy=config.CAMERA_F[1],
                                       cx=config.CAMERA_C[0], cy=config.CAMERA_C[1])
    camera_pose = np.eye(4)
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                               innerConeAngle=np.pi / 16.0,
                               outerConeAngle=np.pi / 6.0)
    scene.add(light, pose=camera_pose)
    global_renderer.point_size = point_size
    color, depth = global_renderer.render(scene, flags=RenderFlags.RGBA)
    return color


def mix_render_result_with_image(image, result):
    result = result / 255.
    alpha = np.expand_dims(result[:, :, 3], 0)
    return alpha * np.moveaxis(result[:, :, :3], -1, 0) + image * (1 - alpha)


def render_mesh(coord, faces, image):
    mesh = trimesh.Trimesh(coord, faces)
    res = render_pyrender_mesh(pyrender.Mesh.from_trimesh(mesh))
    return mix_render_result_with_image(image, res)


def render_pointcloud(coord, image):
    res = render_pyrender_mesh(pyrender.Mesh.from_points(coord), point_size=3.)
    return mix_render_result_with_image(image, res)


def visualize_reconstruction(gt_coord, coord, faces, image):
    gt_pointcloud = render_pointcloud(gt_coord, image)
    pointcloud = render_pointcloud(coord, image)
    mesh = render_mesh(coord, faces, image)
    return np.concatenate((image, gt_pointcloud, pointcloud, mesh), 2)


def p2m_batch_visualize(batch_input, batch_output, faces, atmost=2):
    """
    Every thing is tensor for now, needs to move to cpu and convert to numpy
    """
    batch_size = min(batch_input["images"].size(0), atmost)
    images_stack = []
    for i in range(batch_size):
        image = batch_input["images"][i].cpu().numpy()
        gt_points = batch_input["points"][i].cpu().numpy()
        for j in range(3):
            for k in ["pred_coord_before_deform", "pred_coord"]:
                coord = batch_output[k][j][i].cpu().numpy()
                images_stack.append(visualize_reconstruction(gt_points, coord, faces[j].cpu().numpy(), image))
    return torch.from_numpy(np.concatenate(images_stack, 1))
