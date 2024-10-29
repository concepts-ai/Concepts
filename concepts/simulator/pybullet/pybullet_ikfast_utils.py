#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pybullet_ikfast_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/04/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
from typing import List

from concepts.simulator.pybullet.world import BulletWorld
from concepts.simulator.ikfast.ikfast_common import IKFastWrapperBase


class IKFastPyBulletWrapper(IKFastWrapperBase):
    def __init__(
        self,
        world: BulletWorld, module,
        body_id, joint_ids: List[int], free_joint_ids: List[int] = tuple(),
        use_xyzw: bool = True,  # PyBullet uses xyzw.
        max_attempts: int = 1000,
        fix_free_joint_positions: bool = False,
        shuffle_solutions: bool = False,
        sort_closest_solution: bool = False
    ):
        self.world = world
        self.module = module
        self.body_id = body_id

        joint_info = [self.world.get_joint_info_by_id(self.body_id, joint_id) for joint_id in joint_ids]
        joints_lower = np.array([info.joint_lower_limit for info in joint_info])
        joints_upper = np.array([info.joint_upper_limit for info in joint_info])

        super().__init__(
            module, joint_ids, free_joint_ids,
            joints_lower, joints_upper,
            use_xyzw, max_attempts,
            fix_free_joint_positions, shuffle_solutions, sort_closest_solution
        )

        # assert len(self.free_joint_ids) + 6 == len(self.joint_ids)

    def get_current_joint_positions(self) -> np.ndarray:
        return np.array([self.world.get_joint_state_by_id(self.body_id, joint_id).position for joint_id in self.joint_ids])

    def get_current_free_joint_positions(self) -> np.ndarray:
        return np.array([self.world.get_joint_state_by_id(self.body_id, joint_id).position for joint_id in self.free_joint_ids])
