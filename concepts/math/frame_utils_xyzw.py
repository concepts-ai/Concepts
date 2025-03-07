#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : frame_utils_xyzw.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/18/2020
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Utilities for frame transformations in the XYZW convention. It focuses on transformation matrices and quaternions used in robotics."""

import numpy as np
from typing import Tuple, List, TYPE_CHECKING
from concepts.math.rotationlib_xyzw import quat_mul, quat_conjugate, rotate_vector, mat2quat, quat2mat, axisangle2quat
from concepts.utils.typing_utils import Vec3f, Vec4f

if TYPE_CHECKING:
    from concepts.simulator.pybullet.client import BulletClient


__all__ = [
    'calc_transformation_matrix_from_plane_equation',
    'mat2posquat', 'posquat2mat',
    'compose_transformation', 'inverse_transformation', 'get_transform_a_to_b',
    'frame_mul', 'frame_inv',
    'solve_tool_from_ee', 'solve_ee_from_tool',
    'calc_ee_rotation_mat_from_directions', 'calc_ee_quat_from_directions',
]


def calc_transformation_matrix_from_plane_equation(a, b, c, d):
    """Compute the transformation matrix from a plane equation.

    Args:
        a, b, c, d: the plane equation parameters.

    Returns:
        a 4x4 transformation matrix.
    """
    # Normal of the plane
    N = np.array([a, b, c])
    # Normalize the normal vector
    n = N / np.linalg.norm(N)
    # Z axis
    z_axis = np.array([0, 0, 1])
    # Axis around which to rotate - cross product between Z axis and n
    axis = np.cross(z_axis, n)
    axis = axis / np.linalg.norm(axis)  # normalize the axis
    # Angle between Z axis and n
    angle = np.arccos(np.dot(z_axis, n))
    # Compute rotation matrix using scipy's Rotation
    rotation_matrix = quat2mat(axisangle2quat(axis, angle))
    # Compute translation vector
    translation_vector = -d * n
    # Construct transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    return transformation_matrix


def mat2posquat(mat: np.ndarray) -> Tuple[Vec3f, Vec4f]:
    """Convert a 4x4 transformation matrix to position and quaternion.

    The quaternion is represented as a 4-element array in the order of (x, y, z, w), which follows the convention of pybullet.
    """
    mat = np.asarray(mat, dtype=np.float64)
    pos = mat[:3, 3]
    quat = mat2quat(mat[:3, :3])
    return pos, quat


def posquat2mat(pos: Vec3f, quat: Vec4f) -> np.ndarray:
    """Convert position and quaternion to a 4x4 transformation matrix.

    The quaternion is represented as a 4-element array in the order of (x, y, z, w), which follows the convention of pybullet.
    """
    pos = np.asarray(pos, dtype=np.float64)
    quat = np.asarray(quat, dtype=np.float64)
    mat = np.eye(4)
    mat[:3, :3] = quat2mat(quat)
    mat[:3, 3] = pos
    return mat


def compose_transformation(pos1: Vec3f, quat1: Vec4f, pos2: Vec3f, quat2: Vec4f) -> Tuple[np.ndarray, np.ndarray]:
    """Compose two transformations.

    The transformations are represented as a tuple of position and quaternion.
    The quaternion is represented as a 4-element array in the order of (x, y, z, w), which follows the convention of pybullet.

    Args:
        pos1: the position of the first transformation.
        quat1: the quaternion of the first transformation.
        pos2: the position of the second transformation.
        quat2: the quaternion of the second transformation.

    Returns:
        the composed transformation, represented as a tuple of position and quaternion.
    """
    pos1 = np.asarray(pos1, dtype=np.float64)
    quat1 = np.asarray(quat1, dtype=np.float64)
    pos2 = np.asarray(pos2, dtype=np.float64)
    quat2 = np.asarray(quat2, dtype=np.float64)

    return pos1 + rotate_vector(pos2, quat1), quat_mul(quat1, quat2)


def compose_transformations(pos_quat_list: List[Tuple[Vec3f, Vec4f]]) -> Tuple[np.ndarray, np.ndarray]:
    rv_pos, rv_quat = pos_quat_list[0]
    for (pos_next, quat_next) in pos_quat_list[1:]:
        rv_pos, rv_quat = compose_transformation(rv_pos, rv_quat, pos_next, quat_next)
    return rv_pos, rv_quat


def inverse_transformation(pos: Vec3f, quat: Vec4f) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse a transformation.

    The transformation is represented as a tuple of position and quaternion.
    The quaternion is represented as a 4-element array in the order of (x, y, z, w), which follows the convention of pybullet.
    """
    pos = np.asarray(pos, dtype=np.float64)
    quat = np.asarray(quat, dtype=np.float64)

    inv_pos = rotate_vector(-pos, quat)
    inv_quat = quat_conjugate(quat)
    return inv_pos, inv_quat


def get_transform_a_to_b(pos1: Vec3f, quat1: Vec4f, pos2: Vec3f, quat2: Vec4f) -> Tuple[np.ndarray, np.ndarray]:
    """Get the transformation from frame A to frame B.

    The transformations are represented as a tuple of position and quaternion.
    The quaternion is represented as a 4-element array in the order of (x, y, z, w), which follows the convention of pybullet.
    """
    pos1 = np.asarray(pos1, dtype=np.float64)
    quat1 = np.asarray(quat1, dtype=np.float64)
    pos2 = np.asarray(pos2, dtype=np.float64)
    quat2 = np.asarray(quat2, dtype=np.float64)

    inv_quat1 = quat_conjugate(quat1)
    a_to_b_pos = rotate_vector(pos2 - pos1, inv_quat1)
    a_to_b_quat = quat_mul(inv_quat1, quat2)
    return a_to_b_pos, a_to_b_quat


def frame_mul(pos_a: Vec3f, quat_a: Vec4f, a_to_b: Tuple[Vec3f, Vec4f]) -> Tuple[np.ndarray, np.ndarray]:
    """Multiply a frame with a transformation.

    The frame is represented as a tuple of position and quaternion.
    The transformation is represented as a tuple of position and quaternion.
    The quaternion is represented as a 4-element array in the order of (x, y, z, w), which follows the convention of pybullet.
    """
    pos_a = np.asarray(pos_a, dtype=np.float64)
    quat_a = np.asarray(quat_a, dtype=np.float64)
    transform_pos = np.asarray(a_to_b[0], dtype=np.float64)
    transform_quat = np.asarray(a_to_b[1], dtype=np.float64)

    pos_b = pos_a + rotate_vector(transform_pos, quat_a)
    quat_b = quat_mul(quat_a, transform_quat)
    return pos_b, quat_b


def frame_inv(pos_b: Vec3f, quat_b: Vec4f, a_to_b: Tuple[Vec3f, Vec4f]) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse a frame with a transformation.

    The frame is represented as a tuple of position and quaternion.
    The transformation is represented as a tuple of position and quaternion.
    The quaternion is represented as a 4-element array in the order of (x, y, z, w), which follows the convention of pybullet.
    """
    pos_b = np.asarray(pos_b, dtype=np.float64)
    quat_b = np.asarray(quat_b, dtype=np.float64)
    transform_pos = np.asarray(a_to_b[0], dtype=np.float64)
    transform_quat = np.asarray(a_to_b[1], dtype=np.float64)

    inv_transform_quat = quat_conjugate(transform_quat)
    quat_a = quat_mul(quat_b, inv_transform_quat)
    pos_a = pos_b - rotate_vector(transform_pos, quat_a)
    return pos_a, quat_a


def solve_tool_from_ee(ee_pos: Vec3f, ee_quat: Vec4f, ee_to_tool: Tuple[Vec3f, Vec4f]) -> Tuple[np.ndarray, np.ndarray]:
    return frame_mul(ee_pos, ee_quat, ee_to_tool)


def solve_ee_from_tool(target_tool_pos: Vec3f, target_tool_quat: Vec4f, ee_to_tool: Tuple[Vec3f, Vec4f]) -> Tuple[np.ndarray, np.ndarray]:
    """Solve for the end-effector position and orientation given the tool position and orientation."""
    return frame_inv(target_tool_pos, target_tool_quat, ee_to_tool)


def calc_ee_rotation_mat_from_directions(u: Vec3f, v: Vec3f) -> np.ndarray:
    """Compute the rotation matrix from two directions (the "down" direction for the end effector and the "forward" direction for the end effector).

    Args:
        u: The "down" direction for the end effector.
        v: The "forward" direction for the end effector.
    """
    u = np.asarray(u)
    u = u / np.linalg.norm(u)
    v = np.asarray(v)
    v = v - np.dot(u, v) * u
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)
    return np.array([u, v, w]).T


def calc_ee_quat_from_directions(u: Vec3f, v: Vec3f, default_quat: Vec4f = (0, 1, 0, 0)) -> np.ndarray:
    """Compute the quaternion from two directions (the "down" direction for the end effector and the "forward" direction for the end effector).

    Args:
        u: the "down" direction for the end effector.
        v: the "forward" direction for the end effector.
        default_quat: the default quaternion to be multiplied. This is the quaternion that represents the rotation for the default end effector orientation,
            facing downwards and the forward direction is along the x-axis.
    """
    mat = calc_ee_rotation_mat_from_directions(u, v)
    mat_reference = calc_ee_rotation_mat_from_directions((0, 0, -1), (1, 0, 0))
    quat = mat2quat(np.dot(mat, np.linalg.inv(mat_reference)))
    return quat_mul(quat, default_quat)

