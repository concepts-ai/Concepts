#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : device.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/12/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""
Wrapper around pyrealsense2. Based on:
https://github.com/IntelRealSense/librealsense/issues/8388#issuecomment-782395443
"""

from typing import Dict, List, Optional, Tuple, ClassVar, Callable

import numpy as np
import pyrealsense2 as rs

__all__ = ['RealSenseDevice', 'start_pipelines', 'stop_pipelines']


# Note: change these to fit your use case. Assuming USB 3.2 connection.
_NAME_TO_STREAM_CONFIGURATIONS: Dict[str, List[Tuple]] = {
    # Mapping of camera name to a list of streams to enable
    # in the cfg.enable_stream format
    "Intel RealSense D435": [
        (rs.stream.depth, 1280, 720, rs.format.z16, 30),
        (rs.stream.color, 1920, 1080, rs.format.bgr8, 30),
    ],
    "Intel RealSense L515": [
        (rs.stream.depth, 1024, 768, rs.format.z16, 30),
        (rs.stream.color, 1920, 1080, rs.format.bgr8, 30),
    ],
}


class RealSenseDevice(object):
    align: ClassVar[Callable] = rs.align(rs.stream.color)
    ctx: ClassVar[rs.context] = rs.context()

    def __init__(self, name: str, serial_number: str):
        self.name = name
        self.serial_number = serial_number

        self.pipeline: Optional[rs.pipeline] = None
        self.registered_points: List[Tuple[float, float]] = list()
        self.color_intrinsics = None
        self.depth_intrinsics = None

    @classmethod
    def from_rs_device(cls, dev: rs.device) -> 'RealSenseDevice':
        name = dev.get_info(rs.camera_info.name)
        serial_number = dev.get_info(rs.camera_info.serial_number)
        return cls(name, serial_number)

    @classmethod
    def find_devices(cls, device_filter: str = "") -> List['RealSenseDevice']:
        """
        Get devices as detected by RealSense and filter devices that only
        contain the provided device_filter string in their name.
        e.g. to filter for D435 only you can call `find_devices("D435")`
        """
        devices = [cls.from_rs_device(dev) for dev in cls.ctx.devices]

        # Filter devices
        if device_filter:
            device_filter = device_filter.lower()
            devices = [d for d in devices if device_filter in d.name.lower()]
            print(f"Found devices (filter={device_filter}): {devices}")
        else:
            print(f"Found devices: {devices}")

        if not devices:
            raise RuntimeError("No devices connected!")
        return devices

    @property
    def stream_configurations(self) -> List[Tuple]:
        if self.name not in _NAME_TO_STREAM_CONFIGURATIONS:
            raise RuntimeError(f"Configuration not specified for {self.name}")

        return _NAME_TO_STREAM_CONFIGURATIONS[self.name]

    def start_pipeline(self) -> None:
        """Start RealSense pipeline"""
        if self.pipeline is not None:
            print(f"Pipeline already started for {self}")
            return

        # Setup pipeline and configuration
        pipeline = rs.pipeline(self.ctx)
        cfg = rs.config()
        cfg.enable_device(self.serial_number)
        for stream_configuration in self.stream_configurations:
            cfg.enable_stream(*stream_configuration)

        try:
            prof = pipeline.start(cfg)

            depth_stream = prof.get_stream(rs.stream.depth)
            color_stream = prof.get_stream(rs.stream.color)
            depth_intr = depth_stream.as_video_stream_profile().get_intrinsics()
            color_intr = color_stream.as_video_stream_profile().get_intrinsics()

            self.depth_intrinsics = get_intrinsics_matrix(depth_intr)
            self.color_intrinsics = get_intrinsics_matrix(color_intr)
        except RuntimeError as e:
            message = str(e)
            if message == "Couldn't resolve requests":
                # Something wrong with stream configurations probably
                raise RuntimeError(
                    f"{message} for {self}. Check stream configuration and USB connection."
                )
            else:
                raise e

        # Set the pipeline on the device
        self.pipeline = pipeline
        print(f"Started pipeline for {self}")

    def stop_pipeline(self) -> None:
        """Stop RealSense pipeline"""
        if not self.pipeline:
            print(f"Warning! Device {self} does not have a pipeline initialized")
        else:
            self.pipeline.stop()
            print(f"Stopped pipeline for {self}")

    def capture_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """Capture color and depth images"""
        if self.pipeline is None:
            raise RuntimeError(f"Pipeline for {self} not started!")

        # Get frames and align
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        # Get color and depth frame
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("Could not capture both color and depth frame.")

        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image

    def capture_color_image(self) -> np.ndarray:
        return self.capture_images()[0]

    def capture_depth_image(self) -> np.ndarray:
        return self.capture_images()[1]

    def __str__(self) -> str:
        return f"{self.name} ({self.serial_number})"

    def __repr__(self) -> str:
        return str(self)


def start_pipelines(devices: List[RealSenseDevice]) -> None:
    """Enable each device by starting a stream"""
    for device in devices:
        device.start_pipeline()


def stop_pipelines(devices: List[RealSenseDevice]) -> None:
    """Stop all the pipelines"""
    for device in devices:
        device.stop_pipeline()


def get_intrinsics_matrix(intr) -> np.ndarray:
    fx = float(intr.fx)
    fy = float(intr.fy)
    ppx = float(intr.ppx)
    ppy = float(intr.ppy)
    axs = 0.0
    return np.array([
        [fx, axs, ppx],
        [0.0, fy, ppy],
        [0.0, 0.0, 1.0]
    ])

