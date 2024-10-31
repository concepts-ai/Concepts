#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : device_f.py
# Author : Xiaolin Fang
# Email  : xiaolinf@mit.edu
# Date   : 09/11/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""RealSense interface based on Xiaolin Fang's code"""

import numpy as np
from typing import Tuple

try:
    import pyrealsense2 as rs
    rs_imported = True
except ImportError:
    import unittest.mock
    rs = unittest.mock.Mock()
    rs_imported = False

from concepts.hw_interface.realsense.device import RealSenseInterface


def rs_intrinsics_to_opencv_intrinsics(intr):
    D = np.array(intr.coeffs)
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]])
    return K, D


def get_serial_number(pipeline_profile):
    return pipeline_profile.get_device().get_info(rs.camera_info.serial_number)


def get_intrinsics(pipeline_profile, stream=None):
    if stream is None:
        stream = rs.stream.color
    stream_profile = pipeline_profile.get_stream(stream) # Fetch stream profile for depth stream
    intr = stream_profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
    return rs_intrinsics_to_opencv_intrinsics(intr)


class CaptureRS(RealSenseInterface):

    def __init__(self, callback=None, vis=False, serial_number=None, intrinsics=None, min_tags=1, auto_close=False):
        if not rs_imported:
            raise ImportError('pyrealsense2 is not installed. Please install it using `pip install pyrealsense2`')

        self.callback = callback
        self.vis = vis
        self.min_tags = min_tags
        self.init_serial_number = serial_number

        # Configure depth and color streams
        # 1280 x 720 is the highest realsense dep can get
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial_number is not None:
            config.enable_device(serial_number)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 15)

        # Start streaming
        pipeline_profile = self.pipeline.start(config)

        # And get the device info
        self.serial_number = get_serial_number(pipeline_profile)
        print(f'Connected to {self.init_serial_number}')

        # get the camera intrinsics
        if intrinsics is None:
            self.intrinsics = get_intrinsics(pipeline_profile)
        else:
            self.intrinsics = intrinsics

        if auto_close:
            def close_rs():
                print(f'Closing RealSense camera: {self.init_serial_number}')
                self.close()
            import atexit
            atexit.register(close_rs)

    def get_serial_number(self) -> str:
        return self.serial_number

    def get_rgbd_image(self, format: str = 'rgb') -> Tuple[np.ndarray, np.ndarray]:
        rgb, depth = self.capture()
        if format == 'bgr':
            return rgb[..., ::-1], depth
        elif format == 'rgb':
            return rgb, depth
        else:
            raise ValueError(f'Invalid format: {format}.')

    def get_color_intrinsics(self) -> np.ndarray:
        return self.intrinsics

    def get_depth_intrinsics(self) -> np.ndarray:
        return self.intrinsics

    def capture(self, dep_only=False, to_bgr=False):
        # for _ in range(10):
        # # Wait for a coherent pair of frames: depth and color
        while True:
            frameset = self.pipeline.wait_for_frames(timeout_ms=5000)

            align = rs.align(rs.stream.color)
            frameset = align.process(frameset)

            # Update color and depth frames:
            aligned_depth_frame = frameset.get_depth_frame()
            depth_image = np.asanyarray(aligned_depth_frame.get_data()).copy()
            color_image = np.asanyarray(frameset.get_color_frame().get_data()).copy()
            if dep_only:
                # depth_frame = frameset.get_depth_frame()
                # if not depth_frame: continue
                # depth_image = np.asanyarray(depth_frame.get_data())
                return depth_image

            if to_bgr:
                color_image = color_image[:, :, ::-1]
            return color_image, depth_image

    def skip_frames(self, n):
        for i in range(n):
            _ = self.pipeline.wait_for_frames()

    def close(self):
        self.pipeline.stop()
