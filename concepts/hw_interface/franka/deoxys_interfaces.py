#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : deoxys_interfaces.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/11/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp

from typing import Union, Sequence, Dict
from functools import lru_cache

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from concepts.hw_interface.realsense.device_f import CaptureRS, CaptureRSSubscriber
from concepts.hw_interface.k4a.device import K4ADevice

__all__ = [
    'get_global_config',
    'get_all_setup_names', 'get_setup_config', 'get_default_setup', 'get_default_setup_no_camera',
    'get_robot_config_by_index', 'get_camera_config_by_name',
    'get_franka_interface', 'get_franka_interface_dict',
    'get_realsense_capture', 'get_realsense_capture_dict',
    'get_k4a_capture'
]


@lru_cache(maxsize=1)
def get_global_config():
    return YamlConfig(osp.join(config_root, 'GLOBAL_ENV.yml')).as_easydict()


def get_all_setup_names() -> Sequence[str]:
    return list(get_global_config().setups.keys())


def get_default_setup():
    return get_global_config().setup_defaults.default


def get_default_setup_no_camera():
    return get_global_config().setup_defaults.default_nocam


def get_setup_config(setup_name: str):
    config = get_global_config().setups
    if setup_name in config:
        return config[setup_name]
    raise ValueError(f'Setup with name {setup_name} not found')


def get_robot_config_by_index(robot_index):
    config = get_global_config().robots
    for robot in config:
        if robot.index == robot_index:
            return robot

    raise ValueError(f'Robot with index {robot_index} not found')


def get_robot_config_content_by_index(robot_index):
    return YamlConfig(osp.join(config_root, get_robot_config_by_index(robot_index).config_file)).as_easydict()


def get_camera_config_by_name(camera_name):
    config = get_global_config().cameras
    for camera in config:
        if camera.name == camera_name:
            return camera

    raise ValueError(f'Camera with name {camera_name} not found')


def get_franka_interface(robot_index: int = 1, wait_for_state: bool = True, auto_close: bool = False):
    config_file = get_robot_config_by_index(robot_index).config_file
    robot_interface = FrankaInterface(osp.join(config_root, config_file), use_visualizer=False, auto_close=auto_close)

    if wait_for_state:
        robot_interface.wait_for_state()

    return robot_interface


def get_franka_interface_dict(robots: Sequence[int], wait_for_state: bool = True, auto_close: bool = False) -> Dict[int, FrankaInterface]:
    interfaces = {}
    for robot_index in robots:
        interfaces[robot_index] = get_franka_interface(robot_index, wait_for_state=wait_for_state, auto_close=auto_close)
    return interfaces


def get_realsense_capture(camera_name: str, auto_close: bool = False, skip_frames: int = 0, subscriber: bool = False) -> Union[CaptureRS, CaptureRSSubscriber]:
    camera = get_camera_config_by_name(camera_name)
    if CaptureRSSubscriber.has_camera_publisher(camera_name) or subscriber:
        if camera.get('pub_port') is not None:
            capture = CaptureRSSubscriber(
                host=camera.get('pub_host', None),
                port=camera.pub_port,
                identifier=camera_name,
            )
        else:
            capture = CaptureRSSubscriber.from_identifier(camera_name)
    else:
        capture = CaptureRS(serial_number=str(camera.serial), intrinsics=None, auto_close=auto_close, pub_port=camera.get('pub_port', None))
        if skip_frames > 0:
            capture.skip_frames(skip_frames)
    return capture


def get_realsense_capture_dict(cameras: Sequence[str], auto_close: bool = False, skip_frames: int = 0, subscriber: bool = False) -> Dict[str, Union[CaptureRS, CaptureRSSubscriber]]:
    captures = {}
    for camera_name in cameras:
        captures[camera_name] = get_realsense_capture(camera_name, auto_close=auto_close, skip_frames=skip_frames, subscriber=subscriber)
    return captures


def get_k4a_capture(camera_name: str, skip_frames: int = 0) -> 'K4ADevice':
    assert camera_name == 'mount_k4a'
    device = K4ADevice()
    if skip_frames > 0:
        for _ in range(skip_frames):
            device.capture()
    return device
