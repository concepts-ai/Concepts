#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : base_cartesian_line_controller.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/29/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Control the base of an object following a straight line in the Cartesian space. The object will maintain its orientation during the movement."""

import numpy as np

from sapien.core import Actor
from concepts.math.rotationlib_wxyz import quat_diff_in_axis_angle


class BaseCartesianLineProportionalController(object):
    def __init__(
       self,
       actor: Actor,
       initial_position: np.ndarray,
       target_position: np.ndarray,
       num_steps: int,
       gain: float,
    ):
        self.actor = actor
        self.initial_position = initial_position
        self.target_position = target_position
        self.num_steps = num_steps
        self.gain = gain
        self.velocity_by_step = (self.target_position - self.initial_position) / self.num_steps
        self.target_quat_wxyz = actor.get_pose().q

    def get_projection_t(self, position: np.ndarray) -> float:
        return np.dot((position - self.initial_position), self.velocity_by_step) / np.sum(self.velocity_by_step**2)

    def get_desired_velocity(self) -> np.ndarray:
        position = self.actor.get_pose().p
        t = self.get_projection_t(position)
        target_t = np.maximum(0, np.minimum(t + 1, self.num_steps))
        current_target_position = self.initial_position + target_t * self.velocity_by_step
        velocity = self.gain * (current_target_position - position)
        return velocity

    def get_desired_angular_velocity(self) -> np.ndarray:
        current_quat_wxyz = self.actor.get_pose().q
        return self.gain * quat_diff_in_axis_angle(self.target_quat_wxyz, current_quat_wxyz)

    def set_velocities(self):
        self.actor.set_velocity(self.get_desired_velocity())
        self.actor.set_angular_velocity(self.get_desired_angular_velocity())

    def get_current_t(self) -> float:
        return self.get_projection_t(self.actor.get_pose().p)
