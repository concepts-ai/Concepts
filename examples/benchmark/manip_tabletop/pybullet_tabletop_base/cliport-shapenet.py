#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cliport-shapenet.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/08/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp
import matplotlib.pyplot as plt
from concepts.math.rotationlib_xyzw import rpy
from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.pybullet.camera import TopDownOracle
from concepts.benchmark.manip_tabletop.pybullet_tabletop_base.pybullet_tabletop import TableTopEnv

client = BulletClient(is_gui=True)
env = TableTopEnv(client)

env.initialize_shapenet_core_loader(osp.expanduser('~/Workspace/datasets/ShapeNetCore.v2'))

with client.disable_rendering(suppress_stdout=True):
    env.add_cliport_workspace()
    env.add_shape_net_core_object('laptop', '10f18b49ae496b0109eaabd919821b8', (0.4, 0.0, 0.1), scale=0.3, quat=rpy(90, 90, 0))
    env.set_default_debug_camera()

# Step the simulation for 120 steps to let the objects fall down.
client.step(120)

camera_config = TopDownOracle.get_configs()[0]
rgb, depth, segmentation = client.world.render_image(camera_config, image_size=(320, 640))

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

