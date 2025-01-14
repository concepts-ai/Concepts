#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : franka_base.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/06/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional, Union, Tuple, List

import numpy as np
import torch
import jacinle

import concepts.dm.crow as crow
from concepts.dm.crowhat.impl.pybullet.pybullet_sim_interfaces import PyBulletSimulationControllerInterface
from concepts.simulator.pybullet.components.panda.panda_robot import PandaRobot


logger = jacinle.get_logger(__file__)


class RobotQPosPath(object):
    def __init__(self, path: List[np.ndarray]):
        self.path = path

    def __str__(self):
        if len(self.path) == 0:
            return 'RobotQPosPath(empty)'
        return f'RobotQPosPath(len={len(self.path)}, start={self.path[0]}, end={self.path[-1]})'

    def __repr__(self):
        return super().__str__()

    @classmethod
    def from_interpolation(cls, start: np.ndarray, end: np.ndarray, n: int):
        path = [start + (end - start) * i / n for i in range(n)]
        return cls(path + [end])


def register_state_getter_functions(cogman):
    if not cogman.is_simulation_available():
        return

    env = cogman.get_simulation_env()

    def get_state(state):
        state.batch_set_value('pose_of', torch.tensor(np.array([
            env.world.get_body_state_by_id(obj_id).get_7dpose() for obj_id in cogman.object_pybullet_ids
        ]), dtype=torch.float32))

        return state

    cogman.register_state_getter_function(get_state)


def register_simulation_controllers(cogman, sci: PyBulletSimulationControllerInterface):
    env = cogman.get_simulation_env()

    def move_ctl(trajectory: RobotQPosPath):
        robot: PandaRobot = env.robots[0]
        trajectory = trajectory.path
        last_qpos = np.array(trajectory[-1])
        assert len(last_qpos) in [7, 8], 'Invalid qpos length: {}'.format(len(last_qpos))

        print('Executing trajectory with length: {}'.format(len(trajectory)))
        robot.set_qpos_with_attached_objects(last_qpos[:7])

        if len(last_qpos) == 8:
            print('Unknown gripper state. Please inspect and set the gripper state manually.')
            robot.internal_set_gripper_state(last_qpos[7] > 0.5)

    def grasp_ctl(x):
        robot: PandaRobot = env.robots[0]
        name_mapping = cogman.get_pybullet_name_mapping()
        name2id = name_mapping.name2id

        x_index = name2id[x]
        robot.internal_set_gripper_state(True, body_index=x_index)

    def open_gripper_ctl():
        robot: PandaRobot = env.robots[0]
        if (index := robot.get_attached_object()) is not None:
            current_state = robot.world.get_body_state_by_id(index)
            robot.world.set_body_state2_by_id(
                index,
                position=np.array(current_state.position) - [0, 0, 0.03],
                orientation=np.array(current_state.orientation),
            )
        robot.internal_set_gripper_state(False)

    def close_gripper_ctl():
        robot: PandaRobot = env.robots[0]
        robot.internal_set_gripper_state(True)

    sci.register_controller('move_ctl', move_ctl)
    sci.register_controller('grasp_ctl', grasp_ctl)
    sci.register_controller('open_gripper_ctl', open_gripper_ctl)
    sci.register_controller('close_gripper_ctl', close_gripper_ctl)


def register_physical_controllers(cogman, pci: crow.CrowPhysicalControllerInterface):
    pb_env = cogman.get_simulation_env()
    pb_robot: PandaRobot = pb_env.robot

    def move_ctl(traj: RobotQPosPath):
        pb_robot.move_qpos_path_v2(traj.path)

    def open_ctl(additional_interpolation_multiplier: int = 1):
        pb_robot.open_gripper_free()

    def close_ctl(addtional_interpolation_multiplier: int = 1):
        pb_robot.grasp()

    pci.register_controller('move_ctl', move_ctl)
    pci.register_controller('open_gripper_ctl', open_ctl)
    pci.register_controller('close_gripper_ctl', close_ctl)
    pci.register_controller('grasp_ctl', close_ctl)

    return pci
