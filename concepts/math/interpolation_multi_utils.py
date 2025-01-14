#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : interpolation_multi_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/14/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Scripts for interpolating a trajectory by a multiplier."""

from typing import Optional, Union, Sequence, Tuple, List
from scipy.interpolate import CubicSpline, interp1d
from concepts.math.rotationlib_xyzw import slerp_xyzw, mat2pos_quat_xyzw, pos_quat2mat_xyzw
from concepts.math.rotationlib_wxyz import slerp_wxyz

import numpy as np

__all__ = [
    'interpolate_cubic_spline_multi',
    'interpolate_linear_spline_multi',
    'interpolate_quat_trajectory_multi',
    'interpolate_pose_trajectory_multi',
    'interpolate_posemat_trajectory_multi'
]


def interpolate_cubic_spline_multi(values: Union[np.ndarray, Sequence[np.ndarray]], multiplier: int, xs: Optional[Union[np.ndarray, Sequence[float]]] = None, return_indices: bool = False) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[float]]]:
    """Interpolate a sequence of values by a multiplier by fitting a cubic spline."""
    if xs is None:
        xs = np.arange(len(values))

    cs = CubicSpline(xs, values)
    output_values = [values[0]]
    output_indices = [xs[0]]
    for i in range(0, len(values) - 1):
        for j in range(1, 1 + multiplier):
            new_x = xs[i] + j * (xs[i + 1] - xs[i]) / multiplier
            output_values.append(cs(new_x))
            output_indices.append(new_x)

    assert len(output_values) == (len(values) - 1) * multiplier + 1
    if return_indices:
        return output_values, output_indices
    return output_values


def interpolate_linear_spline_multi(values: Union[np.ndarray, Sequence[np.ndarray]], multiplier: int, xs: Optional[Union[np.ndarray, Sequence[float]]] = None, return_indices: bool = False) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[float]]]:
    """Interpolate a sequence of values by a multiplier using linear interpolation."""
    if xs is None:
        xs = np.arange(len(values))

    ls = interp1d(xs, values, axis=0)
    output_values = [values[0]]
    output_indices = [xs[0]]
    for i in range(0, len(values) - 1):
        for j in range(1, 1 + multiplier):
            new_x = xs[i] + j * (xs[i + 1] - xs[i]) / multiplier
            output_values.append(ls(new_x))
            output_indices.append(new_x)

    assert len(output_values) == (len(values) - 1) * multiplier + 1
    if return_indices:
        return output_values, output_indices
    return output_values


def get_slerp_func(encoding: str = 'xyzw'):
    if encoding == 'xyzw':
        return slerp_xyzw
    elif encoding == 'wxyz':
        return slerp_wxyz
    else:
        raise ValueError(f'Invalid encoding: {encoding}')


def interpolate_quat_trajectory_multi(values: Union[np.ndarray, Sequence[np.ndarray]], multiplier: int, xs: Optional[Union[np.ndarray, Sequence[float]]] = None, encoding: str = 'xyzw', return_indices: bool = False) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[float]]]:
    """Interpolate a quaternion trajectory by a multiplier using spherical linear interpolation."""

    if xs is None:
        xs = np.arange(len(values))

    slerp_func = get_slerp_func(encoding)
    output_values = [values[0]]
    output_indices = [xs[0]]
    for i in range(0, len(values) - 1):
        for j in range(1, 1 + multiplier):
            new_x = j * (xs[i + 1] - xs[i]) / multiplier
            output_values.append(slerp_func(values[i], values[i + 1], new_x))
            output_indices.append(xs[i] + new_x)

    assert len(output_values) == (len(values) - 1) * multiplier + 1
    if return_indices:
        return output_values, output_indices
    return output_values


def interpolate_pose_trajectory_multi(values: Sequence[Tuple[np.ndarray, np.ndarray]], multiplier: int, xs: Optional[Union[np.ndarray, Sequence[float]]] = None, encoding: str = 'xyzw', return_indices: bool = False) -> Union[List[Tuple[np.ndarray, np.ndarray]], Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]]:
    """Interpolate a pose trajectory by a multiplier using spherical linear interpolation."""
    if xs is None:
        xs = np.arange(len(values))

    slerp_func = get_slerp_func(encoding)
    output_values = [values[0]]
    output_indices = [xs[0]]
    for i in range(0, len(values) - 1):
        prev_pos, prev_quat = values[i]
        next_pos, next_quat = values[i + 1]
        for j in range(1, 1 + multiplier):
            new_x = j * (xs[i + 1] - xs[i]) / multiplier
            new_pos = prev_pos + (next_pos - prev_pos) * new_x
            new_quat = slerp_func(prev_quat, next_quat, new_x)
            output_values.append((new_pos, new_quat))
            output_indices.append(xs[i] + new_x)

    assert len(output_values) == (len(values) - 1) * multiplier + 1
    if return_indices:
        return output_values, output_indices
    return output_values


def interpolate_posemat_trajectory_multi(values: Union[np.ndarray, Sequence[np.ndarray]], multiplier: int, xs: Optional[Union[np.ndarray, Sequence[float]]] = None, return_indices: bool = False) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[float]]]:
    """Interpolate a pose matrix trajectory by a multiplier using spherical linear interpolation."""

    if xs is None:
        xs = np.arange(len(values))

    output_values = [values[0]]
    output_indices = [xs[0]]
    for i in range(0, len(values) - 1):
        prev_pos, prev_quat = mat2pos_quat_xyzw(values[i])
        next_pos, next_quat = mat2pos_quat_xyzw(values[i + 1])

        for j in range(1, 1 + multiplier):
            new_x = j * (xs[i + 1] - xs[i]) / multiplier
            new_pos = prev_pos + (next_pos - prev_pos) * new_x
            new_quat = slerp_xyzw(prev_quat, next_quat, new_x)
            new_posemat = pos_quat2mat_xyzw(new_pos, new_quat)
            output_values.append(new_posemat)
            output_indices.append(xs[i] + new_x)

    assert len(output_values) == (len(values) - 1) * multiplier + 1
    if return_indices:
        return output_values, output_indices
    return output_values


