#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : simple_render.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/25/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp
import numpy as np
from typing import Any, Optional, List, Dict

INSIDE_BLENDER = True
try:
    import bpy
except ImportError as e:
    print('Failed to import bpy (Blender Python API).', e)
    INSIDE_BLENDER = False


BASE_DIR = osp.dirname(__file__)


def _set_rendering_engine(
    use_eevee: bool = False, use_gpu: bool = False, render_timeout: Optional[float] = None
):
    if use_eevee:
        bpy.data.scenes["Scene"].render.engine = 'BLENDER_EEVEE'
    else:
        bpy.data.scenes["Scene"].render.engine = 'CYCLES'
        bpy.data.scenes["Scene"].cycles.device = 'GPU' if use_gpu else 'CPU'
        if render_timeout is not None:
            bpy.data.scenes["Scene"].cycles.time_limit = render_timeout


def _set_camera_pos(camera_pos: Optional[List[float]] = None, camera_angle: Optional[List[float]] = None):
    if camera_pos is not None:
        bpy.data.objects['Camera'].location = camera_pos
    if camera_angle is not None:
        bpy.data.objects['Camera'].rotation_euler = camera_angle


def render_scene_simple(
    spec: List[Dict[str, Any]], scene_blend: str,
    output_image: str = 'render.png', output_blendfile: Optional[str] = None,
    camera_pos: Optional[List[float]] = None, camera_angle: Optional[List[float]] = None,
    use_eevee: bool = False, use_gpu: bool = False, render_timeout: Optional[float] = None
):
    """Render a scene with the given specification.

    Args:
        spec: the specification of the scene. Each element is a dictionary with the following keys:
            - filename: the path to the .obj file.
            - scale: the scale of the object.
            - transform: the 4x4 transformation matrix of the object.
        scene_blend: the path to the scene blend file.
        camera_pos: the position of the camera. If None, the default camera position will be used. Format: (x, y, z).
        camera_angle: the angle of the camera. If None, the default camera angle will be used. Format: rotation_euler (x, y, z).
        output_image: the path to the output image.
        output_blendfile: the path to the output blend file.
        use_eevee: whether to use eevee as the rendering engine. It is faster but less accurate.
        use_gpu: whether to use GPU as the rendering device. This is only valid when use_eevee is False.
        render_timeout: the timeout for rendering (in seconds). This is only valid when use_eevee is False.
    """
    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=scene_blend)
    bpy.data.scenes["Scene"].render.filepath = output_image

    _set_camera_pos(camera_pos, camera_angle)
    _set_rendering_engine(use_eevee, use_gpu, render_timeout)

    # Now make some random objects
    for object_dict in spec:
        filename = object_dict['filename']
        scale = object_dict['scale']
        transformation = object_dict['transform']
        # Load the object from .obj file
        bpy.ops.wm.obj_import(filepath=filename, global_scale=scale)

        np_transformation = np.array(transformation).reshape((4, 4))
        # print(np_transformation)

        bpy.context.object.matrix_world = transformation
        # NB(Jiayuan Mao @ 2024/01/25): for some reason, Blender does not update the location of the object.
        bpy.context.object.location = np_transformation[:3, 3]

    while True:
        try:
            bpy.ops.render.render(write_still=True)
            break
        except Exception as e:
            print(e)

    if output_blendfile is not None:
        bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)


def render_scene_incremental_with_new_camera(
    output_image: str = 'render.png',
    camera_pos: Optional[List[float]] = None, camera_angle: Optional[List[float]] = None,
    use_eevee: bool = False, use_gpu: bool = False, render_timeout: Optional[float] = None
):
    bpy.data.scenes["Scene"].render.filepath = output_image
    _set_camera_pos(camera_pos, camera_angle)
    _set_rendering_engine(use_eevee, use_gpu, render_timeout)

    while True:
        try:
            bpy.ops.render.render(write_still=True)
            break
        except Exception as e:
            print(e)
