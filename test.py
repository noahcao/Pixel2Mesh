import os
import bpy
import sys
import math
import random
import numpy as np
from glob import glob
import io
from contextlib import redirect_stdout
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))

light_num_lowbound = 0
light_num_highbound = 6
light_dist_lowbound = 8
light_dist_highbound = 20

g_syn_light_num_lowbound = 0
g_syn_light_num_highbound = 6
g_syn_light_dist_lowbound = 8
g_syn_light_dist_highbound = 20
g_syn_light_azimuth_degree_lowbound = 0
g_syn_light_azimuth_degree_highbound = 0
g_syn_light_elevation_degree_lowbound = -10
g_syn_light_elevation_degree_highbound = 10
g_syn_light_energy_mean = 2
g_syn_light_energy_std = 2
g_syn_light_environment_energy_lowbound = 0
g_syn_light_environment_energy_highbound = 1


def camPosToQuaternion(cx, cy, cz):
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    axis = (-cz, 0, cx)
    angle = math.acos(cy)
    a = math.sqrt(2) / 2
    b = math.sqrt(2) / 2
    w1 = axis[0]
    w2 = axis[1]
    w3 = axis[2]
    c = math.cos(angle / 2)
    d = math.sin(angle / 2)
    q1 = a * c - b * d * w1
    q2 = b * c + a * d * w1
    q3 = a * d * w2 + b * d * w3
    q4 = -b * d * w2 + a * d * w3
    return (q1, q2, q3, q4)


def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return (q1, q2, q3, q4)


def camPosToQuaternion(cx, cy, cz):
    q1a = 0
    q1b = 0
    q1c = math.sqrt(2) / 2
    q1d = math.sqrt(2) / 2
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    t = math.sqrt(cx * cx + cy * cy)
    tx = cx / t
    ty = cy / t
    yaw = math.acos(ty)
    if tx > 0:
        yaw = 2 * math.pi - yaw
    pitch = 0
    tmp = min(max(tx * cx + ty * cy, -1), 1)
    # roll = math.acos(tx * cx + ty * cy)
    roll = math.acos(tmp)
    if cz < 0:
        roll = -roll
    print("%f %f %f" % (yaw, pitch, roll))
    q2a, q2b, q2c, q2d = quaternionFromYawPitchRoll(yaw, pitch, roll)
    q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
    q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
    q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
    return (q1, q2, q3, q4)


def camRotQuaternion(cx, cy, cz, theta):
    theta = theta / 180.0 * math.pi
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = -cx / camDist
    cy = -cy / camDist
    cz = -cz / camDist
    q1 = math.cos(theta * 0.5)
    q2 = -cx * math.sin(theta * 0.5)
    q3 = -cy * math.sin(theta * 0.5)
    q4 = -cz * math.sin(theta * 0.5)
    return (q1, q2, q3, q4)


def quaternionProduct(qx, qy):
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e
    return (q1, q2, q3, q4)


def obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)


num = int(sys.argv[-3])
model = sys.argv[-2]
target = sys.argv[-1]

shape_file = model
setter = model.split('/')
# setting vandom viewpoints for caputing objects
view_params = [[random.uniform(0., 1.) * 360., random.uniform(25, 30.0), 0, 1.25] for i in range(num)]
bpy.ops.import_scene.obj(filepath=shape_file)

# if no texture is provided, we apply a random colour
for o in bpy.context.selected_objects:
    if len(o.data.materials) == 0:
        newMat = bpy.data.materials.new('newmaterial')
        bpy.data.materials.append(newMat)
        r, g, b = (random.uniform(.2, .7), random.uniform(.2, .7), random.uniform(.2, .7))
        o.active_material.diffuse_color = (r, g, b)

bpy.context.scene.render.image_settings.color_mode = 'RGB'
cam = bpy.data.objects['Camera']
bpy.ops.object.select_all(action='TOGGLE')
if 'Lamp' in list(bpy.data.objects.keys()):
    bpy.data.objects['Lamp'].select = True
bpy.ops.object.delete()
for object in bpy.data.objects:
    if object.type == 'MESH':
        for edge in object.data.edges:
            edge.use_freestyle_mark = True
            object.data.show_freestyle_edge_marks = True

# cycling over each picture
f = open(target + 'rendering_metadata.txt', 'w+')
for line in view_params:
    f.write(str(line[0]) + ' ' + str(line[1]) + ' 0 ' + str(line[3]) + '\n')
from mathutils import Matrix
from math import tan

for e, param in enumerate(view_params):

    result = target + str(e) + '.png'
    azimuth_deg = param[0]
    elevation_deg = param[1]
    theta_deg = param[2]
    rho = param[3]

    bpy.ops.object.select_by_type(type='LAMP')
    bpy.ops.object.delete(use_global=False)

    bpy.context.scene.world.light_settings.use_environment_light = True
    bpy.context.scene.world.light_settings.environment_energy = .2
    bpy.context.scene.world.light_settings.environment_color = 'PLAIN'

    bpy.data.worlds['World'].horizon_color = (1, 1, 1)

    # setting random lights
    for i in range(8):
        light_azimuth_deg = i * 360 / 5
        light_elevation_deg = np.random.uniform(g_syn_light_elevation_degree_lowbound,
                                                g_syn_light_elevation_degree_highbound)
        light_dist = 10
        lx, ly, lz = obj_centened_camera_pos(light_dist, light_azimuth_deg, light_elevation_deg)
        bpy.ops.object.lamp_add(type='POINT', view_align=False, location=(lx, ly, lz))
        bpy.data.objects['Point'].data.energy = np.random.normal(.7, .71)

    cx, cy, cz = obj_centened_camera_pos(rho, azimuth_deg, elevation_deg)
    q1 = camPosToQuaternion(cx, cy, cz)
    q2 = camRotQuaternion(cx, cy, cz, theta_deg)
    q = quaternionProduct(q2, q1)
    cam.location[0] = cx
    cam.location[1] = cy
    cam.location[2] = cz
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion[0] = q[0]
    cam.rotation_quaternion[1] = q[1]
    cam.rotation_quaternion[2] = q[2]
    cam.rotation_quaternion[3] = q[3]
    theta_deg = (-1 * theta_deg) % 360
    bpy.data.scenes['Scene'].render.filepath = result

    scene = bpy.context.scene
    scene.render.resolution_x = 224
    scene.render.resolution_y = 224
    scene.render.resolution_percentage = 100
    scene.update()

    # here we save the camera intrinsics
    mat = Matrix([[1., 0., 0., 0.], [0., -0., -1., 0.], [0., 1., -0., 0.], [0., 0., 0., 1.]])

    model_view = (
            cam.matrix_world.inverted() *
            mat
    )

    width = scene.render.resolution_x
    height = scene.render.resolution_y
    aspect_ratio = width / height

    n = cam.data.clip_start
    f = cam.data.clip_end
    fov = cam.data.angle

    proj = Matrix()
    proj[0][0] = 1 / tan(fov / 2)
    proj[1][1] = aspect_ratio / tan(fov / 2)
    proj[2][2] = -(f + n) / (f - n)
    proj[2][3] = - 2 * f * n / (f - n)
    proj[3][2] = - 1
    proj[3][3] = 0

    clip = proj * model_view
    np.save(target + str(e), clip)

    bpy.ops.render.render(write_still=True)