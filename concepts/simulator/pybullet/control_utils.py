#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : control_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/09/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
from typing import Optional
from jacinle.utils.argument import get_nd_shape

from concepts.math.rotationlib_xyzw import quat_diff_in_axis_angle
from concepts.utils.typing_utils import Vec3f, Vec4f


def get_default_joint_pd_control_parameters(nr_joints: int):
    """Return the default parameters for the joint-space PD control."""
    KP = np.asarray([2000.] * nr_joints)
    KD = 2 * np.sqrt(KP)
    config = {
        'kP': KP,
        'kD': KD,
        'max_torque': np.asarray([100.] * nr_joints)
    }
    return config


def get_joint_explicit_pd_control_command(q: np.ndarray, dq: np.ndarray, target_q: np.ndarray, target_dq: Optional[np.ndarray] = None, config: Optional[dict] = None):
    """Joint-space PD control. It uses goal joint states from the feedback thread and current robot states from the subscribed messages to compute
    joint torques.

    Args:
        q: current joint positions.
        dq: current joint velocities.
        target_q: goal joint positions.
        target_dq: goal joint velocities.
        config: control parameters.
    """

    if target_dq is None:
        target_dq = np.zeros_like(dq)

    if config is None:
        config = get_default_joint_pd_control_parameters(q.shape[0])

    delta_q = target_q - q
    delta_dq = target_dq - dq

    # Desired joint torques using PD law
    tau = config['kP'].dot(delta_q) + config['kD'].dot(delta_dq)
    tau = np.clip(tau, -config['max_torque'], config['max_torque'])

    return tau


def get_default_os_imp_control_parameters(kp_pos: float = 20, kp_ori: float = 1, kd_pos: Optional[float] = None, kd_ori: Optional[float] = 0.01, damping_scale: float = 1.0):
    """Return the default parameters for the operation-space impedance control."""
    KP_P = np.asarray(get_nd_shape(kp_pos, 3, type=float))
    KP_O = np.asarray(get_nd_shape(kp_ori, 3, type=float))
    config = {
        'P_pos': KP_P,
        'D_pos': 2 * damping_scale * np.sqrt(KP_P) if kd_pos is None else np.asarray(get_nd_shape(kd_pos, 3, type=float)),
        'P_ori': KP_O,
        'D_ori': 2 * damping_scale * np.sqrt(KP_O) if kd_ori is None else np.asarray(get_nd_shape(kd_ori, 3, type=float)),
    }
    return config


def get_os_imp_control_command(curr_pos: Vec3f, curr_quat: Vec4f, target_pos: Vec3f, target_quat: Vec4f, curr_vel: Vec3f, curr_omg: Vec3f, J: np.ndarray, config: Optional[dict] = None):
    """Operation-space impedance control. It uses goal pose from the feedback thread and current robot states from the subscribed messages to compute
    task-space force, and then the corresponding joint torques.

    Implementation is based on: https://github.com/justagist/pybullet_robot/blob/master/src/pybullet_robot/controllers/os_impedance_ctrl.py
    Also reference: https://github.com/NVIDIA-Omniverse/orbit/blob/57e766cf68c942191a74e24269b780ee9a817535/source/extensions/omni.isaac.orbit/omni/isaac/orbit/controllers/operational_space.py#L307

    Args:
        curr_pos: current end-effector position.
        curr_quat: current end-effector orientation.
        target_pos: goal end-effector position.
        target_quat: goal end-effector orientation.
        curr_vel: current end-effector velocity.
        curr_omg: current end-effector angular velocity.
        J: end-effector Jacobian.
        config: control parameters.

    Returns:
        joint torques.
    """

    delta_pos = (target_pos - curr_pos)
    delta_ori = quat_diff_in_axis_angle(target_quat, curr_quat)

    # print self._goal_pos, curr_pos
    # Desired task-space force using PD law
    F_p = np.concatenate([config['P_pos'] * delta_pos, config['P_ori'] * delta_ori])
    F_d = np.concatenate([config['D_pos'] * curr_vel, config['D_ori'] * curr_omg])
    F = F_p - F_d  # Equivalent to F = Kp * (x_d - x) + Kd * (0 - x_dot)

    return np.dot(J.T, F).flatten()


def get_os_imp_control_command_robot(robot, target_pos: Vec3f, target_quat: Vec4f, config: Optional[dict] = None):
    """Alias of get_os_imp_control_command, but with a robot instance as the input.

    Args:
        robot: robot instance. The robot should have implemented the following methods:
            get_ee_pose: return the current end-effector pose.
            get_ee_velocity: return the current end-effector velocity.
            get_jacobian: return the current end-effector Jacobian.
        target_pos: goal end-effector position.
        target_quat: goal end-effector orientation.
        config: control parameters.
    """
    curr_pos, curr_quat = robot.get_ee_pose()
    curr_vel, curr_omg = robot.get_ee_velocity()
    J = robot.get_jacobian()
    return get_os_imp_control_command(curr_pos, curr_quat, target_pos, target_quat, curr_vel, curr_omg, J, config)


def get_default_joint_imp_control_parameters():
    """Return the default parameters for the joint-space impedance control."""
    KP = np.asarray([2000., 2000., 2000., 2000., 2000., 2000., 2000.])
    KD = 2 * np.sqrt(KP)
    config = {
        'P_joint': KP,
        'D_joint': KD,
    }
    return config


def get_joint_imp_control_command(q: np.ndarray, dq: np.ndarray, target_q: np.ndarray, target_dq: Optional[np.ndarray] = None, config: Optional[dict] = None):
    """Joint-space impedance control. It uses goal joint states from the feedback thread and current robot states from the subscribed messages to compute
    joint torques.

    Implementation is based on a simplified version of:
    https://github.com/NVIDIA-Omniverse/orbit/blob/57e766cf68c942191a74e24269b780ee9a817535/source/extensions/omni.isaac.orbit/omni/isaac/orbit/controllers/joint_impedance.py

    Args:
        q: current joint positions.
        dq: current joint velocities.
        target_q: goal joint positions.
        target_dq: goal joint velocities.
        config: control parameters.
    """

    if target_dq is None:
        target_dq = np.zeros_like(dq)

    if config is None:
        config = get_default_joint_imp_control_parameters()

    delta_q = target_q - q
    delta_dq = target_dq - dq

    # Desired joint torques using PD law
    tau_p = config['P_joint'].dot(delta_q)
    tau_d = config['D_joint'].dot(delta_dq)
    tau = tau_p + tau_d

    return tau