#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pose_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/23/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
from typing import Optional, Union, Tuple

from concepts.utils.typing_utils import VecNf, Vec3f, Vec4f
from concepts.math.rotationlib_xyzw import quat_diff


def canonicalize_pose(pos: Union[Vec3f, Tuple[Vec3f, Vec4f]], quat: Optional[Vec4f] = None) -> Tuple[np.ndarray, np.ndarray]:
    assert (
        len(pos) == 3 and (quat is not None and len(quat) == 4) or
        len(pos) == 2 and len(pos[0]) == 3 and len(pos[1]) == 4 and quat is None
    ), f'Invalid input: pos={pos}, quat={quat}'

    if len(pos) == 3 and quat is not None:
        pos = np.asarray(pos)
        quat = np.asarray(quat)
    elif len(pos) == 2:
        pos, quat = np.asarray(pos[0]), np.asarray(pos[1])
    else:
        raise RuntimeError(f'Invalid input: pos={pos}, quat={quat}')

    return pos, quat


def pose_difference(pose1: Tuple[Vec3f, Vec4f], pose2: Tuple[Vec3f, Vec4f]) -> np.ndarray:
    """Compute the difference between two poses: `pose2 - pose1`."""
    pos1, quat1 = pose1
    pos2, quat2 = pose2
    axis, angle = quat_diff(quat2, quat1, return_axis=True)
    return np.concatenate([np.asarray(pos2) - np.asarray(pos1), axis * angle])


def pose_distance2(pose1: Tuple[Vec3f, Vec4f], pose2: Tuple[Vec3f, Vec4f]) -> Tuple[float, float]:
    """Compute the difference between two poses: `||pose2 - pose1||`. This functino returns the positional and angular distance."""
    pos1, quat1 = pose1
    pos2, quat2 = pose2
    angle = quat_diff(quat2, quat1)
    return np.linalg.norm(np.asarray(pos2) - np.asarray(pos1)), abs(angle)


def pose_distance(pose1: Tuple[Vec3f, Vec4f], pose2: Tuple[Vec3f, Vec4f]) -> float:
    """Compute the difference between two poses: `||pose2 - pose1||`."""
    pos_error, angle_error = pose_distance2(pose1, pose2)
    return pos_error + angle_error


def angle_distance(quat1: Vec4f, quat2: Vec4f) -> float:
    """Compute the difference between two quaternions: `||quat2 - quat1||`."""
    return quat_diff(quat1, quat2)

