import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
from pyrender import RenderFlags

import config
from utils.mesh import Ellipsoid

ellipsoid = Ellipsoid()
mesh = trimesh.Trimesh(vertices=ellipsoid.coord, faces=ellipsoid.faces[0])
# print(mesh.vertices)

# m = pyrender.Mesh.from_trimesh(mesh)

m = pyrender.Mesh.from_points(mesh.vertices)
scene = pyrender.Scene(ambient_light=[.03, .03, .03],
                       bg_color=[0., 0., 0., 0.])
scene.add(m)
camera = pyrender.IntrinsicsCamera(fx=config.CAMERA_F[0], fy=config.CAMERA_F[1],
                                   cx=config.CAMERA_C[0], cy=config.CAMERA_C[1])
camera_pose = np.eye(4)
scene.add(camera, pose=camera_pose)
light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                           innerConeAngle=np.pi / 16.0,
                           outerConeAngle=np.pi / 6.0)
scene.add(light, pose=camera_pose)
r = pyrender.OffscreenRenderer(config.CAMERA_RES[0], config.CAMERA_RES[1], point_size=3.)
color, depth = r.render(scene, flags=RenderFlags.RGBA)
print(color.shape)
print(color[:3, :3])
plt.imshow(color)
plt.show()