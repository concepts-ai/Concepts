#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : interpolation_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/26/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional, Union, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline as _CubicSpline
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from concepts.math.rotationlib_wxyz import slerp as slerp_wxyz, quat_diff as quat_diff_wxyz
from concepts.math.rotationlib_xyzw import slerp as slerp_xyzw, quat_diff as quat_diff_xyzw

__all__ = [
    'SplineInterface', 'CubicSpline', 'LinearSpline', 'SlerpSpline', 'PoseSpline',
    'gen_cubic_spline', 'gen_linear_spline', 'project_to_cubic_spline', 'get_next_target_cubic_spline', 'project_to_linear_spline', 'get_next_target_linear_spline'
]

class SplineInterface(object):
    def get_min_x(self) -> float:
        raise NotImplementedError()

    def get_max_x(self) -> float:
        raise NotImplementedError()

    def __call__(self, x: float) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        raise NotImplementedError()

    def derivative(self, n: int = 1, x: Optional[float] = None) -> Union['SplineInterface', Tuple['SplineInterface', ...]]:
        raise NotImplementedError('Derivative is not implemented.')

    def project_to(self, y: Union[np.ndarray, Tuple[np.ndarray, ...]], minimum_x: Optional[float] = None) -> float:
        raise NotImplementedError('Project is not implemented.')

    def get_next(self, y: Union[np.ndarray, Tuple[np.ndarray, ...]], step_size: float, minimum_x: Optional[float] = None) -> Tuple[float, Union[np.ndarray, Tuple[np.ndarray, ...]]]:
        """Get the next target point on a spline interpolation.

        Args:
            y: the current point.
            step_size: the step size.
            minimum_x: the minimum x value to be considered.

        Returns:
            the next x-value and the next target point.
        """
        x = self.project_to(y, minimum_x=minimum_x)
        x = min(max(x + step_size, self.get_min_x()), self.get_max_x())
        return x, self(x)


class CubicSpline(_CubicSpline, SplineInterface):
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        self.xs = np.asarray(xs, dtype=np.float32)
        self.ys = np.asarray(ys, dtype=np.float32)
        super(CubicSpline, self).__init__(self.xs, self.ys, axis=0)

    @classmethod
    def from_points(cls, ys: np.ndarray) -> 'CubicSpline':
        xs = np.arange(len(ys))
        return cls(xs, ys)

    def get_min_x(self) -> float:
        return float(self.xs[0])

    def get_max_x(self) -> float:
        return float(self.xs[-1])

    def project_to(self, y: np.ndarray, minimum_x: Optional[float] = None) -> float:
        """Project a point to a cubic spline interpolation.

        Args:
            y: the point to be projected.
            minimum_x: the minimum x value to be considered.

        Returns:
            the time of the projected point.
        """
        y = np.asarray(y, dtype=np.float32)
        dslp = self.derivative()

        def f(x):
            return ((self(float(x)) - y) ** 2).sum()

        def df(x):
            return 2 * ((self(float(x)) - y) * dslp(float(x))).sum()

        x0 = np.argmin(np.abs(self.ys - y).sum(axis=-1))
        min_x = max(self.get_min_x(), x0 - 1)
        if minimum_x is not None:
            min_x = max(min_x, minimum_x)
        res = minimize(f, x0, jac=df, method='L-BFGS-B', bounds=[min_x, max(min_x + 0.1, min(self.get_max_x(), x0 + 1))])
        return float(res.x[0])


class LinearSpline(SplineInterface):
    """Linear spline interpolation.

    This class is a wrapper of scipy.interpolate.interp1d that mimics the CubicSpline interface.
    """
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        assert len(xs) == len(ys)
        self.xs = np.asarray(xs, dtype=np.float32)
        self.ys = np.asarray(ys, dtype=np.float32)
        assert len(self.xs.shape) == 1, 'xs should be a 1D array.'
        self.interpolator = interp1d(xs, ys, axis=0, kind='linear', fill_value='extrapolate')

    @classmethod
    def from_points(cls, ys: np.ndarray) -> 'LinearSpline':
        xs = np.arange(len(ys))
        return cls(xs, ys)

    def get_min_x(self) -> float:
        return float(self.xs[0])

    def get_max_x(self) -> float:
        return float(self.xs[-1])

    def __call__(self, x: float) -> np.ndarray:
        return self.interpolator(x)

    def project_to(self, y: np.ndarray, minimum_x: Optional[float] = None) -> float:
        """Project a point to a linear spline interpolation.

        Args:
            y: the point to be projected.
            minimum_x: the minimum x value to be considered.

        Returns:
            the time of the projected point.
        """
        y = np.asarray(y, dtype=np.float32)

        def f(x):
            y_prime = self(x)
            return ((y - y_prime) ** 2).sum()

        x0 = np.argmin(np.abs(self.ys - y).sum(axis=-1))
        min_x = max(self.get_min_x(), x0 - 1)
        if minimum_x is not None:
            min_x = max(min_x, minimum_x)
        res = minimize(f, x0, bounds=[(min_x, max(min_x + 0.1, min(self.get_max_x(), x0 + 1)))])
        return float(res.x[0])


class SlerpSpline(SplineInterface):
    """Slerp spline interpolation.

    This class is a wrapper of scipy.interpolate.interp1d that mimics the CubicSpline interface.
    """
    def __init__(self, xs: np.ndarray, ys: np.ndarray, quat_format: str = 'xyzw'):
        assert len(xs) == len(ys)
        self.xs = np.asarray(xs, dtype=np.float32)
        self.ys = np.asarray(ys, dtype=np.float32)
        self.quat_format = quat_format
        assert len(self.xs.shape) == 1, 'xs should be a 1D array.'

        if self.quat_format == 'xyzw':
            self.slerp = slerp_xyzw
            self.diff = quat_diff_xyzw
        elif self.quat_format == 'wxyz':
            self.slerp = slerp_wxyz
            self.diff = quat_diff_wxyz
        else:
            raise ValueError('Unknown format: {}'.format(self.quat_format))

        fake_ys = np.arange(len(ys))
        self.interpolator = interp1d(xs, fake_ys, axis=0, kind='linear', fill_value='extrapolate')

    @classmethod
    def from_points(cls, ys: np.ndarray, quat_format: str = 'xyzw') -> 'SlerpSpline':
        xs = np.arange(len(ys))
        return cls(xs, ys, quat_format=quat_format)

    def get_min_x(self) -> float:
        return float(self.xs[0])

    def get_max_x(self) -> float:
        return float(self.xs[-1])

    def __call__(self, x: float) -> np.ndarray:
        y = self.interpolator(x)
        y0 = int(np.floor(y))
        y1 = int(np.ceil(y))

        if y0 == y1:
            return self.ys[y0]
        if y1 <= 0:
            y0, y1 = 0, 1
        if y0 >= len(self.ys) - 1:
            y0, y1 = len(self.ys) - 2, len(self.ys) - 1

        t = (y - y0)
        return self.slerp(self.ys[y0], self.ys[y1], t)

    def project_to(self, y: np.ndarray, minimum_x: Optional[float] = None) -> float:
        """Project a point to a slerp spline interpolation.

        Args:
            y: the point to be projected.
            minimum_x: the minimum x value to be considered.

        Returns:
            the time of the projected point.
        """
        y = np.asarray(y, dtype=np.float32)

        def f(x):
            y_prime = self(x)
            diff = self.diff(y, y_prime)
            return abs(diff)

        x0 = np.argmin(np.abs(self.ys - y).sum(axis=-1))
        min_x = max(self.get_min_x(), x0 - 1)
        if minimum_x is not None:
            min_x = max(min_x, minimum_x)
        res = minimize(f, x0, bounds=[(min_x, max(min_x + 0.1, min(self.get_max_x(), x0 + 1)))])
        return float(res.x[0])


class PoseSpline(SplineInterface):
    """Pose spline interpolation.

    This class is a wrapper of scipy.interpolate.interp1d that mimics the CubicSpline interface.
    """
    def __init__(self, xs: np.ndarray, y_pos: np.ndarray, y_quat: np.ndarray, quat_format: str = 'xyzw', fixed_quat: Optional[bool] = None):
        """Initialize a pose spline interpolation.

        Args:
            xs: the time values.
            y_pos: the position values.
            y_quat: the quaternion values.
            quat_format: the quaternion format ('xyzw' or 'wxyz').
            fixed_quat: whether the quaternion values are the same for all time steps.
        """
        assert len(xs) == len(y_pos) == len(y_quat)
        self.xs = np.asarray(xs, dtype=np.float32)
        self.y_pos = np.asarray(y_pos, dtype=np.float32)
        self.y_quat = np.asarray(y_quat, dtype=np.float32)
        self.pos_spline = LinearSpline(self.xs, self.y_pos)
        self.quat_spline = SlerpSpline(self.xs, self.y_quat, quat_format=quat_format)
        self.mixed_spline = LinearSpline(self.xs, np.concatenate([self.y_pos, self.y_quat], axis=-1))

        self.fixed_quat = fixed_quat
        if self.fixed_quat is None:
            self.fixed_quat = np.allclose(self.y_quat, self.y_quat[0:1])

    @classmethod
    def from_points(cls, y_pos: np.ndarray, y_quat: np.ndarray, quat_format: str = 'xyzw') -> 'PoseSpline':
        xs = np.arange(len(y_pos))
        return cls(xs, y_pos, y_quat, quat_format=quat_format)

    @classmethod
    def from_pose_sequence(cls, pose_sequence: Sequence[Tuple[np.ndarray, np.ndarray]], quat_format: str = 'xyzw') -> 'PoseSpline':
        y_pos = np.asarray([pose[0] for pose in pose_sequence])
        y_quat = np.asarray([pose[1] for pose in pose_sequence])
        return cls.from_points(y_pos, y_quat, quat_format=quat_format)

    def get_min_x(self) -> float:
        return float(self.xs[0])

    def get_max_x(self) -> float:
        return float(self.xs[-1])

    def __call__(self, x: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.fixed_quat:
            return self.pos_spline(x), self.y_quat[0]
        return self.pos_spline(x), self.quat_spline(x)

    def project_to(self, y: Tuple[np.ndarray, np.ndarray], minimum_x: Optional[float] = None) -> float:
        """Project a point to a pose spline interpolation.

        Args:
            y: the point to be projected.
            minimum_x: the minimum x value to be considered.

        Returns:
            the time of the projected point.
        """
        if self.fixed_quat:
            pos = y[0]
            x = self.pos_spline.project_to(pos, minimum_x=minimum_x)
        else:
            pos, quat = y
            x = self.mixed_spline.project_to(np.concatenate([pos, quat], axis=-1), minimum_x=minimum_x)
        return x


"""Below are the obsolete functions that do not use the object-oriented interface."""


def gen_cubic_spline(ys: np.ndarray) -> CubicSpline:
    """Generate a cubic spline interpolation from an array of points."""
    return CubicSpline.from_points(ys)


def gen_linear_spline(ys) -> LinearSpline:
    """Generate a linear spline interpolation from an array of points."""
    return LinearSpline.from_points(ys)


def project_to_cubic_spline(spl: CubicSpline, y: np.ndarray, ys: np.ndarray) -> float:
    """Project a point to a cubic spline interpolation.

    Args:
        spl: the cubic spline interpolation.
        y: the point to be projected.
        ys: the array of points that the cubic spline interpolation is generated from.

    Returns:
        the time of the projected point.
    """
    return spl.project_to(y, minimum_x=0)


def get_next_target_cubic_spline(spl: CubicSpline, y: np.ndarray, step_size: float, ys: np.ndarray) -> Tuple[float, np.ndarray]:
    """Get the next target point on a cubic spline interpolation.

    Args:
        spl: the cubic spline interpolation.
        y: the current point.
        step_size: the step size.
        ys: the array of points that the cubic spline interpolation is generated from.

    Returns:
        the next target point.
    """
    return spl.get_next(y, step_size, minimum_x=0)


def project_to_linear_spline(spl: LinearSpline, y: np.ndarray, minimum_x: Optional[float] = None) -> float:
    """Project a point to a linear spline interpolation.

    Args:
        spl: the linear spline interpolation.
        y: the point to be projected.
        minimum_x: the minimum x value to be considered.

    Returns:
        the time of the projected point.
    """
    return spl.project_to(y, minimum_x=minimum_x)


def get_next_target_linear_spline(spl: LinearSpline, y: np.ndarray, step_size: float, minimum_x: Optional[float] = None) -> Tuple[float, np.ndarray]:
    """Get the next target point on a linear spline interpolation.

    Args:
        spl: the linear spline interpolation.
        y: the current point.
        step_size: the step size.
        minimum_x: the minimum x value to be considered.

    Returns:
        the next target point (time, point).
    """
    return spl.get_next(y, step_size, minimum_x=minimum_x)

