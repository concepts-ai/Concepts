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

import io
import numpy as np
import time
from typing import Optional, Union, Tuple

try:
    import pyrealsense2 as rs
    rs_imported = True
except ImportError:
    import unittest.mock
    rs = unittest.mock.Mock()
    rs_imported = False

import jacinle
from jacinle import get_local_addr
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
    stream_profile = pipeline_profile.get_stream(stream)  # Fetch stream profile for depth stream
    intr = stream_profile.as_video_stream_profile().get_intrinsics()  # Downcast to video_stream_profile and fetch intrinsics
    return rs_intrinsics_to_opencv_intrinsics(intr)


class CaptureRS(RealSenseInterface):
    def __init__(self, callback=None, vis=False, serial_number=None, intrinsics=None, min_tags=1, auto_close=False, pub_port: Optional[int] = None):
        if not rs_imported:
            raise ImportError('pyrealsense2 is not installed. Please install it using `pip install pyrealsense2`')

        self.callback = callback
        self.vis = vis
        self.min_tags = min_tags
        self.init_serial_number = serial_number
        self.pub_port = pub_port

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

    def publish(self, port: Optional[int] = None, max_fps: Optional[int] = None, name: Optional[str] = None, register_name_server: bool = False):
        try:
            import zmq
        except ImportError:
            raise ImportError('zmq is not installed. Please install it using `pip install zmq`')

        if port is None:
            port = self.pub_port

        if register_name_server:
            from jacinle.comm.service_name_server import sns_register
            sns_register(f'concepts/realsense/{name}', {'port': port, 'host': get_local_addr()})

        assert port is not None, 'Port is not specified'
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind(f'tcp://*:{port}')

        try:
            with jacinle.tqdm_pbar() as pbar:
                while True:
                    current_time = time.time()
                    self.publish_once(socket, current_time)
                    current_time_string = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
                    if name is not None:
                        pbar.set_description(f'Capture of [{name}] published at {current_time_string} (port: {port})')
                    else:
                        pbar.set_description(f'Capture published at {current_time_string} (port: {port})')
                    pbar.update()
                    if max_fps is not None:
                        sleep_time = 1 / max_fps - (time.time() - current_time)
                        if sleep_time > 0.05:
                            time.sleep(sleep_time)
        except KeyboardInterrupt:
            print('KeyboardInterrupt: Closing RealSense camera')
            pass

    def publish_once(self, socket, current_time):
        rgb, dep = self.capture()
        intrinsics = self.get_color_intrinsics()[0]
        kwargs = {'color_image': rgb, 'depth_image': dep, 'intrinsics': intrinsics, 'time': current_time}

        # Use np.savez_compressed to save the images and intrinsics
        stream = io.BytesIO()
        np.savez_compressed(stream, **kwargs)
        socket.send(stream.getvalue())

    def close(self):
        self.pipeline.stop()


class CaptureRSSubscriber(object):
    def __init__(self, host: Optional[str] = None, port: Optional[Union[int, str]] = None, identifier: Optional[str] = None):
        if host is None:
            host = 'localhost'

        try:
            import zmq
        except ImportError:
            raise ImportError('zmq is not installed. Please install it using `pip install zmq`')

        context = zmq.Context()
        self.host = host
        self.port = port
        self.identifier = identifier

        identifier = identifier or f'rs@{port}'
        print(f'Connecting to the realsense camera {identifier} at tcp://{host}:{port}... Waiting...')
        self.socket = context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, '')
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect(f'tcp://{host}:{port}')
        self.intrinsics = self._receive_raw()['intrinsics'], None
        print(f'Connected to the realsense camera {identifier}.')

    @classmethod
    def has_camera_publisher(cls, identifier):
        from jacinle.comm.service_name_server import sns_has
        return sns_has(f'concepts/realsense/{identifier}')

    @classmethod
    def from_identifier(cls, identifier):
        from jacinle.comm.service_name_server import sns_get
        info = sns_get(f'concepts/realsense/{identifier}')
        rv = cls(host=info['host'], port=info['port'], identifier=identifier)
        return rv

    def get_human_name(self):
        if self.identifier is not None:
            return self.identifier
        return f'{self.host}:{self.port}'

    def get_color_intrinsics(self):
        return self.intrinsics

    def get_depth_intrinsics(self):
        return self.intrinsics

    def _receive_raw(self):
        data = self.socket.recv()
        stream = io.BytesIO(data)
        kwargs = np.load(stream)
        return kwargs

    def capture(self, return_time: bool = False, return_intrinsics: bool = False):
        assert not return_time or not return_intrinsics
        kwargs = self._receive_raw()
        if return_time:
            return kwargs['color_image'], kwargs['depth_image'], float(kwargs['time'])
        if return_intrinsics:
            return kwargs['color_image'], kwargs['depth_image'], kwargs['intrinsics']
        return kwargs['color_image'], kwargs['depth_image']

