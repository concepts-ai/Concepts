#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cliport-ocrtoc.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/08/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import jacinle
import pybullet as p
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.pybullet.camera import CLIPortCamera, TopDownOracle
from concepts.benchmark.manip_tabletop.pybullet_tabletop_base.pybullet_tabletop import TableTopEnv

client = BulletClient(is_gui=False)
env = TableTopEnv(client)

all_ocrtoc_objects = env.list_ocrtoc_objects()
print(all_ocrtoc_objects)

all_ycb_simplified_objects = env.list_ycb_simplified_objects()
print(all_ycb_simplified_objects)

npr.seed(1)

object_ids = list()
with client.disable_rendering(suppress_stdout=True):
    env.add_cliport_workspace()
    selected_objects = npr.choice(all_ocrtoc_objects, size=3)
    for o in selected_objects:
        pos = (npr.uniform(0.4, 0.7), npr.uniform(-0.3, 0.3), 0.1)
        object_id = env.add_ocrtoc_object(o, pos)
        object_ids.append(object_id)

    selected_objects = npr.choice(all_ycb_simplified_objects, size=2)
    for o in selected_objects:
        pos = (npr.uniform(0.4, 0.7), npr.uniform(-0.3, 0.3), 0.1)
        object_id = env.add_ycb_simplified_object(o, pos)
        object_ids.append(object_id)

    env.set_default_debug_camera()

# Step the simulation for 120 steps to let the objects fall down.
client.step(240)

camera_config = TopDownOracle.get_configs()[0]
rgb, depth, segmentation = client.world.render_image(camera_config, image_size=(320, 640))

camera_config = CLIPortCamera.get_configs()[0]
rgb, depth, segmentation = client.world.render_image(camera_config, image_size=(480, 640))

# show the three images in a row
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(rgb)
plt.subplot(132)
plt.imshow(depth)
plt.subplot(133)
plt.imshow(segmentation)
plt.tight_layout()
plt.show()


def export_scene(client: BulletClient, object_ids: list[int]):
    objects = list()
    for object_id in object_ids:
        mesh_infos = client.world.get_all_mesh_info(object_id)
        assert len(mesh_infos) == 1
        filename, scale, pos, quat = mesh_infos[0]
        rotation_mat = p.getMatrixFromQuaternion(quat)
        rotation_mat = np.array(rotation_mat).reshape(3, 3)
        translation_mat = np.eye(4)
        translation_mat[:3, :3] = rotation_mat
        translation_mat[:3, 3] = pos
        objects.append({
            'filename': filename,
            'scale': scale,
            'transform': translation_mat.tolist()
        })
    return objects


objects = export_scene(client, object_ids)
jacinle.mkdir('./generated')
jacinle.dump('./generated/scene_example1.json', objects)
print('Scene exported to ./generated/scene_example1.json')

