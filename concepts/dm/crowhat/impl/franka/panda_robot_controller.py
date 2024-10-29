#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : panda_robot_controller.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/30/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Tuple

import numpy as np

from concepts.hw_interface.franka.fri_client import FrankaRemoteClient

from concepts.dm.crow.interfaces.controller_interface import CrowPhysicalControllerInterface
from concepts.dm.crowhat.world.manipulator_interface import RobotArmJointTrajectory, SingleArmControllerInterface

__all__ = ['PandaControllerInterface']


class PandaControllerInterface(SingleArmControllerInterface):
    def __init__(self, panda_robot: FrankaRemoteClient):
        super().__init__()
        self._panda_robot = panda_robot

    @property
    def panda_robot(self) -> FrankaRemoteClient:
        return self._panda_robot

    def get_qpos(self) -> np.ndarray:
        return self.panda_robot.get_qpos()

    def get_qvel(self) -> np.ndarray:
        return self.panda_robot.get_qvel()

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.panda_robot.get_ee_pose()

    def move_home(self) -> bool:
        return self.panda_robot.move_home()

    def move_qpos(self, qpos: np.ndarray, **kwargs) -> None:
        self.get_update_default_parameters('move_qpos', kwargs)
        self.panda_robot.move_qpos(qpos, **kwargs)

    def move_qpos_trajectory(self, trajectory: RobotArmJointTrajectory, **kwargs) -> None:
        self.get_update_default_parameters('move_qpos_trajectory', kwargs)
        self.panda_robot.move_qpos_trajectory(trajectory, **kwargs)

    def open_gripper(self, **kwargs) -> None:
        self.get_update_default_parameters('open_gripper', kwargs)
        self.panda_robot.open_gripper(**kwargs)

    def close_gripper(self, **kwargs) -> None:
        self.get_update_default_parameters('close_gripper', kwargs)
        self.panda_robot.close_gripper(**kwargs)

    def grasp(self, width: float = 0.05, force: float = 40, **kwargs) -> None:
        self.get_update_default_parameters('grasp', kwargs)
        self.panda_robot.grasp(width, force, **kwargs)

    def move_grasp(self, approaching_trajectory: RobotArmJointTrajectory, width: float = 0.05, force: float = 40, **kwargs) -> None:
        self.get_update_default_parameters('grasp', kwargs)
        self.move_qpos_trajectory(approaching_trajectory)
        self.grasp(width, force)

    def move_place(self, placing_trajectory: RobotArmJointTrajectory, **kwargs) -> None:
        self.get_update_default_parameters('place', kwargs)
        self.move_qpos_trajectory(placing_trajectory)
        self.open_gripper()

    def attach_physical_interface(self, physical_interface: CrowPhysicalControllerInterface):
        physical_interface.register_controller('move_home', self.move_home)
        physical_interface.register_controller('move_qpos', self.move_qpos)
        physical_interface.register_controller('move_qpos_trajectory', self.move_qpos_trajectory)
        physical_interface.register_controller('open_gripper', self.open_gripper)
        physical_interface.register_controller('close_gripper', self.close_gripper)
        physical_interface.register_controller('grasp', self.grasp)

        physical_interface.register_controller('ctl_grasp', self.move_grasp)
        physical_interface.register_controller('ctl_place', self.move_place)
