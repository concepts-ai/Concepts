#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : rotationlib_wxyz.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/04/2019
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""
Tools for converting between rotation representations.

Conventions
-----------

- All functions accept batches as well as individual rotations.
- All rotation conventions match respective MuJoCo defaults (e.g., quaternions use wxyz convention).
    Note that this is DIFFERENT from PyBullet (which uses xyzw).
- All angles are in radians.
- Matricies follow LR convention.
- Euler Angles are all relative with 'xyz' axes ordering.
- See specific representation for more information.

Representations
---------------

Euler
    There are many euler angle frames -- here we will strive to use the default
    in MuJoCo, which is eulerseq='xyz'.

    This frame is a relative rotating frame, about x, y, and z axes in order.
    Relative rotating means that after we rotate about x, then we use the
    new (rotated) y, and the same for z.

Quaternions
    These are defined in terms of rotation (angle) about a unit vector (x, y, z)
    We use the following <q0, q1, q2, q3> convention:

    .. code-block:: python

        q0 = cos(angle / 2)
        q1 = sin(angle / 2) * x
        q2 = sin(angle / 2) * y
        q3 = sin(angle / 2) * z

    This is also sometimes called qw, qx, qy, qz.

    Note that quaternions are ambiguous, because we can represent a rotation by
    angle about vector <x, y, z> and -angle about vector <-x, -y, -z>.
    To choose between these, we pick "first nonzero positive", where we
    make the first nonzero element of the quaternion positive.

    This can result in mismatches if you're converting an quaternion that is not
    "first nonzero positive" to a different representation and back.

Axis Angle
    .. warning::

        (Not currently implemented)
        These are very straightforward.  Rotation is angle about a unit vector.

XY Axes
    .. warning::

        (Not currently implemented)
        We are given x axis and y axis, and z axis is cross product of x and y.

Z Axis
    .. warning::

        This is NOT RECOMMENDED.  Defines a unit vector for the Z axis,
        but rotation about this axis is not well defined.

    Instead pick a fixed reference direction for another axis (e.g. X)
    and calculate the other (e.g. Y = Z cross-product X),
    then use XY Axes rotation instead.
SO3
    .. warning::

        (Not currently implemented)
        While not supported by MuJoCo, this representation has a lot of nice features.
        We expect to add support for these in the future.

TODOs/Missings
    - Rotation integration or derivatives (e.g. velocity conversions)
    - More representations (SO3, etc)
    - Random sampling (e.g. sample uniform random rotation)
    - Performance benchmarks/measurements
    - (Maybe) define everything as to/from matricies, for simplicity
"""

# Copyright (c) 2009-2017, Matthew Brett and Christoph Gohlke
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Many methods borrow heavily or entirely from transforms3d:
# https://github.com/matthew-brett/transforms3d
# They have mostly been modified to support batched operations.

import itertools
import numpy as np

# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0


def as_rotation(r):
    """Convert a 3x3 matrix or a quaternion into a standard 3x3 matrix representation."""
    r = np.asarray(r, dtype=np.float64)
    if isinstance(r, np.ndarray) and r.shape == (3, 3):
        return r
    if isinstance(r, np.ndarray) and r.shape == (4,):
        return quat2mat(r)
    raise ValueError('Invalid rotation: {}.'.format(r))


def wxyz2xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4
    return np.concatenate([quat[..., 1:], quat[..., :1]], axis=-1)


def xyzw2wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4
    return np.concatenate([quat[..., -1:], quat[..., :-1]], axis=-1)


def rpy(r, p, y, degree=True):
    """Create a quaternion from euler angles."""
    if degree:
        r = np.deg2rad(r)
        p = np.deg2rad(p)
        y = np.deg2rad(y)
    return euler2quat((r, p, y))


def euler2mat(euler, homogeneous: bool = False):
    """ Convert Euler Angles to Rotation Matrix.  See rotation.py for notes """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    if homogeneous:
        mat = np.empty(euler.shape[:-1] + (4, 4), dtype=np.float64)
        mat[..., 3, 3] = 1.0
    else:
        mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat


def euler2quat(euler):
    """ Convert Euler Angles to Quaternions.  See rotation.py for notes """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shape euler {}".format(euler)

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat


def mat2euler(mat):
    """ Convert Rotation Matrix to Euler Angles.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(
        condition, -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]), -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1])
    )
    euler[..., 1] = np.where(condition, -np.arctan2(-mat[..., 0, 2], cy), -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition, -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]), 0.0)
    return euler


def mat2quat(mat):
    """ Convert Rotation Matrix to Quaternion.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    Qxx, Qyx, Qzx = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    Qxy, Qyy, Qzy = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    Qxz, Qyz, Qzz = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(mat.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=['multi_index'])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q


def quat2euler(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    return mat2euler(quat2mat(quat))


def subtract_euler(e1, e2):
    assert e1.shape == e2.shape
    assert e1.shape[-1] == 3
    q1 = euler2quat(e1)
    q2 = euler2quat(e2)
    q_diff = quat_mul(q1, quat_conjugate(q2))
    return quat2euler(q_diff)


def quat2mat(quat, homogeneous: bool = False):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    if homogeneous:
        mat = np.empty(quat.shape[:-1] + (4, 4), dtype=np.float64)
        mat[..., 3, 3] = 1.0
    else:
        mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)

    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)

    if homogeneous:
        return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(4))
    else:
        return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def quat_conjugate(q):
    q = np.asarray(q, dtype=np.float64)
    inv_q = -q
    inv_q[..., 0] *= -1
    return inv_q


def quat_mul(q0, q1, *args):
    """ Multiply two quaternions."""

    if len(args) > 0:
        q = quat_mul(q0, q1)
        for q_i in args:
            q = quat_mul(q, q_i)
        return q

    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)

    assert q0.shape == q1.shape
    assert q0.shape[-1] == 4
    assert q1.shape[-1] == 4

    w0 = q0[..., 0]
    x0 = q0[..., 1]
    y0 = q0[..., 2]
    z0 = q0[..., 3]

    w1 = q1[..., 0]
    x1 = q1[..., 1]
    y1 = q1[..., 2]
    z1 = q1[..., 3]

    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1
    z = w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1
    q = np.array([w, x, y, z])
    if q.ndim == 2:
        q = q.swapaxes(0, 1)
    assert q.shape == q0.shape
    return q


def quat_pow(q, n):
    q = np.asarray(q, dtype=np.float64)
    assert q.shape[-1] == 4

    theta = 0
    sin_theta = np.linalg.norm(q[..., 1:])

    if sin_theta > 0.0001:
        theta = 2 * np.arcsin(sin_theta)
        theta *= 1 if q[..., 0] >= 0 else -1

    theta *= n
    axis = q[..., 1:] / sin_theta
    return axisangle2quat(axis, theta)


def quat_diff(q0, q1, return_axis=False):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q_diff = quat_mul(q0, quat_conjugate(q1))
    axis, angle = quat2axisangle(q_diff)
    if return_axis:
        return axis, angle
    return angle


def quat_diff_in_axis_angle(q0, q1):
    axis, angle = quat_diff(q0, q1, return_axis=True)
    return axis * angle


def quat_rot_vec(q, v0):
    q = np.asarray(q, dtype=np.float64)
    q_v0 = np.array([0, v0[0], v0[1], v0[2]])
    q_v = quat_mul(q, quat_mul(q_v0, quat_conjugate(q)))
    v = q_v[1:]
    return v


def quat_rot_vec_batch(q, v_batch):
    quat = np.asarray(q)
    vec = np.asarray(v_batch)

    u = quat[1:]
    s = quat[0]

    return 2.0 * np.dot(vec, u)[..., np.newaxis] * u + (s * s - np.dot(u, u)) * vec + 2.0 * s * np.cross(u, vec)


def rotate_vector(v, q):
    """Rotate a vector by a quaternion."""
    return quat_rot_vec(q, v)


def rotate_vector_batch(v_batch, q):
    """Rotate a vector by a quaternion."""
    return quat_rot_vec_batch(q, v_batch)


def quat_identity():
    return np.array([1, 0, 0, 0], dtype='float64')


def slerp(q0, q1, t):
    """Spherical linear interpolation between two quaternions.

    .. code-block:: latex

        q(t) = q_0 * (q_0^{-1} * q_1)^t

    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    return quat_mul(q0, quat_pow(quat_mul(quat_conjugate(q0), q1), t))


def axisangle2quat(axis, angle):
    quat = np.zeros(4, dtype='float64')
    quat[0] = np.cos(angle / 2)
    quat[1:] = np.sin(angle / 2) * axis
    return quat


def quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64)
    theta = 0
    axis = np.array([0, 0, 1])
    sin_theta = np.linalg.norm(quat[1:])

    if sin_theta > 0.0001:
        theta = 2 * np.arcsin(sin_theta)
        theta *= 1 if quat[0] >= 0 else -1
        axis = quat[1:] / sin_theta

    return axis, theta


def euler2point_euler(euler):
    euler = np.asarray(euler, dtype=np.float64)

    _euler = euler.copy()
    if len(_euler.shape) < 2:
        _euler = np.expand_dims(_euler, 0)
    assert _euler.shape[1] == 3
    _euler_sin = np.sin(_euler)
    _euler_cos = np.cos(_euler)
    return np.concatenate([_euler_sin, _euler_cos], axis=-1)


def point_euler2euler(euler):
    euler = np.asarray(euler, dtype=np.float64)

    _euler = euler.copy()
    if len(_euler.shape) < 2:
        _euler = np.expand_dims(_euler, 0)
    assert _euler.shape[1] == 6
    angle = np.arctan(_euler[..., :3] / _euler[..., 3:])
    angle[_euler[..., 3:] < 0] += np.pi
    return angle


def quat2point_quat(quat):
    # Should be in qw, qx, qy, qz
    quat = np.asarray(quat, dtype=np.float64)
    _quat = quat.copy()
    if len(_quat.shape) < 2:
        _quat = np.expand_dims(_quat, 0)
    assert _quat.shape[1] == 4
    angle = np.arccos(_quat[:, [0]]) * 2
    xyz = _quat[:, 1:]
    xyz[np.squeeze(np.abs(np.sin(angle / 2))) >= 1e-5] = (xyz / np.sin(angle / 2))[
        np.squeeze(np.abs(np.sin(angle / 2))) >= 1e-5
    ]
    return np.concatenate([np.sin(angle), np.cos(angle), xyz], axis=-1)


def point_quat2quat(quat):
    # Should be in sin(q), cos(q), qx, qy, qz
    quat = np.asarray(quat, dtype=np.float64)
    _quat = quat.copy()
    if len(_quat.shape) < 2:
        _quat = np.expand_dims(_quat, 0)
    assert _quat.shape[1] == 5
    angle = np.arctan(_quat[:, [0]] / _quat[:, [1]])
    qw = np.cos(angle / 2)

    qxyz = _quat[:, 2:]
    qxyz[np.squeeze(np.abs(np.sin(angle / 2))) >= 1e-5] = (qxyz * np.sin(angle / 2))[
        np.squeeze(np.abs(np.sin(angle / 2))) >= 1e-5
    ]
    return np.concatenate([qw, qxyz], axis=-1)


def normalize_angles(angles):
    """Puts angles in [-pi, pi] range."""
    angles = np.asarray(angles, dtype=np.float64)
    angles = angles.copy()
    if angles.size > 0:
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        assert -np.pi - 1e-6 <= angles.min() and angles.max() <= np.pi + 1e-6
    return angles


def round_to_straight_angles(angles):
    """Returns closest angle modulo 90 degrees """
    angles = np.asarray(angles, dtype=np.float64)
    angles = np.round(angles / (np.pi / 2)) * (np.pi / 2)
    return normalize_angles(angles)


def get_parallel_rotations():
    """Returns a list of all possible rotations that are parallel to the canonical axes."""
    mult90 = [0, np.pi / 2, -np.pi / 2, np.pi]
    parallel_rotations = []
    for euler in itertools.product(mult90, repeat=3):
        canonical = mat2euler(euler2mat(euler))
        canonical = np.round(canonical / (np.pi / 2))
        if canonical[0] == -2:
            canonical[0] = 2
        if canonical[2] == -2:
            canonical[2] = 2
        canonical *= np.pi / 2
        if all([(canonical != rot).any() for rot in parallel_rotations]):
            parallel_rotations += [canonical]
    assert len(parallel_rotations) == 24
    return parallel_rotations


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def find_orthogonal_vector(v: np.ndarray) -> np.ndarray:
    """Find an orthogonal vector to the given vector.

    The returned vector is guaranteed to be normalized.
    """
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)

    mask = np.abs(v[..., 0]) < 0.5
    return np.cross(v, np.array([1, 0, 0])) * mask + np.cross(v, np.array([0, 1, 0])) * (1 - mask)


def quaternion_from_axes(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Converts a rotation matrix to a quaternion."""
    m = np.stack([x, y, z], axis=1)
    return mat2quat(m)


def quaternion_from_vectors(vec1, vec2):
    """Create a rotation quaternion q such that q * vec1 = vec2."""
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    assert vec1.shape == vec2.shape, f'Vector shapes do not match: {vec1.shape} != {vec2.shape}'
    assert vec1.shape[-1] == 3, f'Vector shape must be 3, got {vec1.shape[-1]}'
    assert vec2.shape[-1] == 3, f'Vector shape must be 3, got {vec2.shape[-1]}'

    vec1 = vec1 / np.linalg.norm(vec1, axis=-1, keepdims=True)
    vec2 = vec2 / np.linalg.norm(vec2, axis=-1, keepdims=True)

    u = np.cross(vec1, vec2)
    s = np.dot(vec1, vec2)

    if np.linalg.norm(u) < 1e-6:
        return np.array([0, 0, 0, 1])

    opposite_pairs = (s < -1 + 1e-6)
    u = u * (1 - opposite_pairs) + opposite_pairs * find_orthogonal_vector(u)

    if len(u.shape) == 1:
        s = np.array([s], dtype=u.dtype)
    q = np.concatenate([u, s + 1], axis=-1)

    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return q


def enumerate_quaternion_from_vectors(input_normal, target_normal, nr_samples: int = 4):
    base_quat = quaternion_from_vectors(input_normal, target_normal)

    yaw_quat = quaternion_from_vectors(target_normal, np.array([0, 0, 1]))
    for yaw in np.arange(0, 2 * np.pi, 2 * np.pi / nr_samples):
        quat = quat_mul(
            quat_conjugate(yaw_quat),
            rpy(0, 0, yaw, degree=False),
            yaw_quat,
            base_quat
        )
        yield quat


def mat2pos_quat(mat):
    """Convert a 4x4 matrix to a position and quaternion vector."""
    pos = mat[:3, 3]
    quat = mat2quat(mat[:3, :3])
    return pos, quat


def pos_quat2mat(pos, quat):
    """Convert position and quaternion to a 4x4 matrix."""
    pos = np.asarray(pos)
    quat = np.asarray(quat)
    assert pos.shape == (3,)
    assert quat.shape == (4,)

    mat = np.eye(4)
    mat[:3, :3] = quat2mat(quat)
    mat[:3, 3] = pos
    return mat


# For all functions that involve quaternions, we create a copy of the function that ends with _wxyz

as_rotation_wxyz = as_rotation
mat2quat_wxyz = mat2quat
quat2mat_wxyz = quat2mat
quat2euler_wxyz = quat2euler
euler2quat_wxyz = euler2quat
quat_conjugate_wxyz = quat_conjugate
quat_mul_wxyz = quat_mul
quat_pow_wxyz = quat_pow
quat_diff_wxyz = quat_diff
quat_diff_in_axis_angle_wxyz = quat_diff_in_axis_angle
quat_rot_vec_wxyz = quat_rot_vec
quat_rot_vec_batch_wxyz = quat_rot_vec_batch
rotate_vector_wxyz = rotate_vector
rotate_vector_batch_wxyz = rotate_vector_batch
quat_identity_wxyz = quat_identity
slerp_wxyz = slerp
axisangle2quat_wxyz = axisangle2quat
quat2axisangle_wxyz = quat2axisangle
quat_wxyz2point_quat = quat2point_quat
point_quat2quat_wxyz = point_quat2quat
quaternion_from_axes_wxyz = quaternion_from_axes
quaternion_from_vectors_wxyz = quaternion_from_vectors
enumerate_quaternion_from_vectors_wxyz = enumerate_quaternion_from_vectors
mat2pos_quat_wxyz = mat2pos_quat
pos_quat2mat_wxyz = pos_quat2mat
