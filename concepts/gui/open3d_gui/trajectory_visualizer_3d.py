#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : trajectory_visualizer_3d.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/11/2025
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import jacinle
import time
import numpy as np
import open3d as o3d

from concepts.math.rotationlib_xyzw import pos_quat2mat_xyzw


def visualize_object_trajectory_open3d(scene, object_trajectories, dt: float = 0.1, verbose: bool = False, wait_time: float = 0):
    """Visualize the object trajectories in Open3D."""
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    visualize_objects = {name: _clone_and_transform(o, object_trajectories[0].get(name, None)) for name, o in scene.items()}
    current_transforms = {name: object_trajectories[0].get(name, None) for name in visualize_objects}
    for o in visualize_objects.values():
        vis.add_geometry(o)

    start_time = time.time()
    if wait_time > 0:
        while True:
            if time.time() - start_time > wait_time:
                break
            vis.poll_events()
            vis.update_renderer()

    start_time = time.time()
    for step in jacinle.tqdm(object_trajectories, disable=verbose):
        for name, transform in step.items():
            if name not in visualize_objects:
                continue
            _delta_transform(visualize_objects[name], current_transforms[name], transform)
            current_transforms[name] = transform
            vis.update_geometry(visualize_objects[name])

        if verbose:
            print('Updating renderer...', step)

        while True:
            if time.time() - start_time > dt:
                break
            vis.poll_events()
            vis.update_renderer()

        start_time = time.time()

    vis.run()
    vis.destroy_window()


def visualize_object_state_open3d(o3d_objects, state):
    target_objects = list()
    for name, o3d_object in o3d_objects.items():
        if name in state:
            target_objects.append(_clone_and_transform(o3d_object, state[name]))
        else:
            target_objects.append(o3d_object)
    o3d.visualization.draw_geometries(target_objects)  # type: ignore


def _clone_and_transform(o3d_object, transform=None):
    if transform is None:
        return o3d_object
    pos, quat = transform
    transform = pos_quat2mat_xyzw(pos, quat)
    if isinstance(o3d_object, o3d.geometry.PointCloud):
        return o3d.geometry.PointCloud(o3d_object).transform(transform)
    elif isinstance(o3d_object, o3d.geometry.TriangleMesh):
        return o3d.geometry.TriangleMesh(o3d_object).transform(transform)
    else:
        raise ValueError('Unsupported object type: {}'.format(type(o3d_object)))


def _delta_transform(o3d_object, current_transform, target_transform):
    if current_transform is None or target_transform is None:
        return

    current_pos, current_quat = current_transform
    target_pos, target_quat = target_transform

    current_mat = pos_quat2mat_xyzw(current_pos, current_quat)
    target_mat = pos_quat2mat_xyzw(target_pos, target_quat)

    delta_mat = target_mat @ np.linalg.inv(current_mat)
    o3d_object: o3d.geometry.TriangleMesh
    o3d_object.transform(delta_mat)
