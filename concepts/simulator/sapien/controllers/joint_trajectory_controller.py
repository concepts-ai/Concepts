#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : joint_trajectory_controller.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/29/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
from sapien.core import Actor
from concepts.math.rotationlib_wxyz import quat_diff_in_axis_angle


class JointTrajectoryProportionalController(object):
    def __init__(
        self,
        actor: Actor,
        p_array: np.ndarray,
        q_array: np.ndarray,
        gain: float,
        coef: float = 1
    ):
        self.actor = actor
        self.p_array = p_array
        self.q_array = q_array
        self.target_quat_wxyz = actor.get_pose().q
        self.gain = gain
        self.coef = coef
        self.num_steps = len(p_array) - 1
        self.current_t = 0

    def get_projection_t(self, current_p, current_q) -> int:
        current_p = np.array(current_p)
        current_q = np.array(current_q)
        start = max(self.current_t - 5, 0)
        end = min(self.current_t + 6, self.num_steps)
        p_dist = np.sum((self.p_array[start:end] - current_p)**2, axis=1)
        # TODO: find a more reasonable way to calculate q_dist
        q_dist = np.sum((self.q_array[start:end] - current_q)**2, axis=1)
        idx = np.argmin(p_dist + self.coef * q_dist)
        # print(p_dist[idx], q_dist[idx])
        t = idx + start
        return t

    def get_desired_velocities(self, t=None) -> tuple[np.ndarray, np.ndarray]:
        current_p = self.actor.get_pose().p
        current_q = self.actor.get_pose().q
        if t is None:
            t = self.get_projection_t(current_p, current_q)
        self.current_t = t
        target_t = np.maximum(0, np.minimum(t + 1, self.num_steps))
        current_target_position = self.p_array[target_t]
        current_target_orientation = self.q_array[target_t]
        velocity = self.gain * (current_target_position - current_p)
        angular_velocity = self.gain * quat_diff_in_axis_angle(current_target_orientation, current_q)
        return velocity, angular_velocity

    def set_velocities(self, contact=None):
        velocity, angular_velocity = self.get_desired_velocities()
        if contact is not None:
            contact_point = contact['point']
            contact_normal = contact['normal']
            actor_center = self.actor.get_pose().p
            velocity, angular_velocity = self.get_desired_velocities(t=self.current_t + 1)
            vel_contact_point = velocity + _cross(angular_velocity, contact_point - actor_center)
            vel_contact_normal = np.dot(vel_contact_point, contact_normal) * contact_normal
            print(velocity, vel_contact_point, vel_contact_normal)
            velocity = velocity - vel_contact_normal
            print(velocity)
        self.actor.set_velocity(velocity)
        self.actor.set_angular_velocity(angular_velocity)

    def get_current_t(self) -> int:
        return self.get_projection_t(self.actor.get_pose().p, self.actor.get_pose().q)


def _cross(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Workaround for the bug in PyCharm type hinting
    return np.cross(x, y)
