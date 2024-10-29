#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : server.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/11/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from jacinle.logging import get_logger
from jacinle.comm.service import Service, SocketClient

if TYPE_CHECKING:
    import franka_interface
    from concepts.hw_interface.realsense.device import RealSenseDevice

logger = get_logger(__file__)

__all__ = ['FrankaService', 'FrankaServiceClient']


class FrankaService(Service):
    DEFAULT_PORTS = [12345, 12346]

    arm: Optional['franka_interface.ArmInterface']
    camera: Optional['RealSenseDevice']
    camera_extrinsics: Optional[np.ndarray]

    def initialize(self):
        self.arm = None
        self.camera = None
        self.camera_extrinsics = None

        if self.configs.get('debug', False):
            logger.critical('Running in debug mode.')
            return

        try:
            import rospy
            import franka_interface
            rospy.init_node('franka_interface')
            self.arm = franka_interface.ArmInterface()
        except ImportError:
            logger.warning('Failed to import franka_interface. Please install it first.')

        if self.configs.get('camera', None) == 'D435':
            from concepts.hw_interface.realsense.device import RealSenseDevice
            self.camera = RealSenseDevice.find_devices('D435')[0]
            self.camera.start_pipeline()

        if self.configs.get('camera_extrinsics', None) is not None:
            with open(self.configs.get('camera_extrinsics')) as f:
                self.camera_extrinsics = np.asarray(list(map(float, f.read().split())))

    def finalize(self):
        if self.camera is not None:
            self.camera.stop_pipeline()

    @property
    def gripper(self):
        if self.arm is None:
            raise RuntimeError('FrankaInterface is not initialized.')
        return self.arm.hand

    def call(self, command, *args, **kwargs):
        if self.arm is None:
            logger.critical(f'[DEBUG] Received command {command} with args {args}.')
            return

        function = getattr(self, command)
        logger.critical(f'Received command {command}.')
        return function(*args, **kwargs)

    def move_qpos(self, qpos: np.ndarray, timeout: float = 10) -> None:
        """Move the robot to a given joint position."""
        qpos_dict = self.arm.convertToDict(qpos)
        self.arm.move_to_joint_positions(qpos_dict, timeout)

    def move_home(self, timeout: float = 10) -> None:
        """Move the robot to the home position."""
        self.arm.move_to_neutral(timeout)

    def move_qpos_to_touch(self, qpos: np.ndarray, timeout: float = 10) -> None:
        """Move the robot to a touch position."""
        qpos_dict = self.arm.convertToDict(qpos)
        self.arm.move_to_touch(qpos_dict, timeout)

    def move_qpos_from_touch(self, qpos: np.ndarray, timeout: float = 10) -> None:
        """Move the robot from a touch position to a given joint position."""
        qpos_dict = self.arm.convertToDict(qpos)
        self.arm.move_from_touch(qpos_dict, timeout)

    def move_qpos_trajectory(self, qpos_list, timeout: float = 10) -> None:
        qpos_dict = [self.arm.convertToDict(qpos) for qpos in qpos_list]
        self.arm.execute_position_path(qpos_dict, timeout)

    def reset_errors(self) -> None:
        """Reset the error state of the robot."""
        self.arm.resetErrors()

    def open_gripper(self) -> None:
        """Open the gripper."""
        self.gripper.open()

    def close_gripper(self) -> None:
        """Close the gripper."""
        self.gripper.close()

    def grasp_gripper(self, width: float, force: float = 40):
        """Grasp an object with the gripper."""
        self.gripper.grasp(width, force)

    def get_qpos(self) -> np.ndarray:
        """Return the current joint position."""
        return np.array(self.arm.convertToList(self.arm.joint_angles()), dtype=np.float32)

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the end-effector pose."""
        rv = self.arm.endpoint_pose()
        o = rv['orientation']
        return np.array(rv['position'], dtype=np.float32), np.array([o.x, o.y, o.z, o.w], dtype=np.float32)

    def set_cart_impedance_freemotion(self):
        """Set the robot to Cartesian impedance mode with free motion."""
        self.arm.set_cart_impedance_pose(
            self.arm.endpoint_pose(),
            stiffness=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )

    def capture_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the color and depth image."""
        if self.camera is None:
            raise RuntimeError('Camera is not initialized.')
        return self.camera.capture_images()

    def get_camera_intrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the color and depth intrinsics matrices."""
        if self.camera is None:
            raise RuntimeError('Camera is not initialized.')
        return self.camera.color_intrinsics, self.camera.depth_intrinsics

    def get_camera_extrinsics(self) -> np.ndarray:
        """Return the color and depth intrinsics matrices."""
        if self.camera is None:
            raise RuntimeError('Camera is not initialized.')
        return self.camera_extrinsics


class FrankaServiceClient(SocketClient):
    def __init__(self, server: str, name='franka-client', port_pair=None, automatic_reset_errors: bool = True):
        if port_pair is None:
            port_pair = FrankaService.DEFAULT_PORTS

        connection = (f'tcp://{server}:{port_pair[0]}', f'tcp://{server}:{port_pair[1]}')
        super().__init__(name, connection)

        self.automatic_reset_errors = automatic_reset_errors

    def move_qpos(self, qpos: np.ndarray, timeout: float = 10) -> None:
        self.call('move_qpos', qpos, timeout)
        if self.automatic_reset_errors:
            self.call('reset_errors')

    def move_home(self, timeout: float = 10) -> None:
        self.call('move_home', timeout)
        if self.automatic_reset_errors:
            self.call('reset_errors')

    def move_qpos_to_touch(self, qpos: np.ndarray, timeout: float = 10) -> None:
        self.call('move_qpos_to_touch', qpos, timeout)
        if self.automatic_reset_errors:
            self.call('reset_errors')

    def move_qpos_from_touch(self, qpos: np.ndarray, timeout: float = 10) -> None:
        self.call('move_qpos_from_touch', qpos, timeout)
        if self.automatic_reset_errors:
            self.call('reset_errors')

    def move_qpos_trajectory(self, qpos_list, timeout: float = 10) -> None:
        current_qpos = self.get_qpos()
        qpos_list = [current_qpos] + qpos_list
        self.call('move_qpos_trajectory', qpos_list, timeout)
        if self.automatic_reset_errors:
            self.call('reset_errors')

    def reset_errors(self) -> None:
        return self.call('reset_errors')

    def open_gripper(self, width: float = 0.2) -> None:
        self.call('open_gripper', width)
        if self.automatic_reset_errors:
            self.call('reset_errors')

    def close_gripper(self) -> None:
        self.call('close_gripper')
        if self.automatic_reset_errors:
            self.call('reset_errors')

    def grasp_gripper(self, width: float = 0.05, force: float = 40):
        self.call('grasp_gripper', width, force)
        if self.automatic_reset_errors:
            self.call('reset_errors')

    def get_qpos(self) -> np.ndarray:
        return self.call('get_qpos')

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.call('get_ee_pose')

    def set_cart_impedance_freemotion(self):
        return self.call('set_cart_impedance_freemotion')

    def capture_image(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.call('capture_image')

    def get_camera_intrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.call('get_camera_intrinsics')

    def get_camera_extrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.call('get_camera_extrinsics')

