#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : device.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/22/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional
from dataclasses import dataclass
import numpy as np

try:
    import pykinect_azure as k4a
    k4a_imported = True
except ImportError:
    import unittest.mock
    k4a = unittest.mock.Mock()
    k4a_imported = False


@dataclass
class K4ACapture(object):
    rv: bool
    rgb: Optional[np.ndarray] = None
    depth_in_color: Optional[np.ndarray] = None
    xyz_in_color: Optional[np.ndarray] = None


def maybe_init_k4a():
    if not k4a_imported:
        raise ImportError('pykinect_azure is not installed. Please install it using `pip install pykinect_azure`')

    import pykinect_azure.k4a._k4a as _C
    if _C.k4a_dll is None:
        k4a.initialize_libraries()


class K4ADevice(object):
    def __init__(self, config=None):
        if not k4a_imported:
            raise ImportError('pykinect_azure is not installed. Please install it using `pip install pykinect_azure`')

        maybe_init_k4a()

        if config is None:
            config = k4a.default_configuration
            config.color_format = k4a.K4A_IMAGE_FORMAT_COLOR_BGRA32
            config.color_resolution = k4a.K4A_COLOR_RESOLUTION_720P
            config.depth_mode = k4a.K4A_DEPTH_MODE_NFOV_2X2BINNED

        self.device = k4a.start_device(config=config)
        self.intrinsics = self.device.calibration.get_matrix(k4a.K4A_CALIBRATION_TYPE_COLOR)

    def capture(self) -> K4ACapture:
        capture = self.device.update()

        rv, depth_in_color = capture.get_transformed_depth_image()
        if not rv:
            return K4ACapture(rv)

        rv, xyz_in_color = capture.get_transformed_pointcloud()
        if not rv:
            return K4ACapture(rv)

        rv, rgb = capture.get_color_image()
        if not rv:
            return K4ACapture(rv)

        rgb = rgb[..., :3][..., ::-1]

        return K4ACapture(rv, rgb, depth_in_color, xyz_in_color)
