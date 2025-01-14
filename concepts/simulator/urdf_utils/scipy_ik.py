#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : scipy_ik.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/18/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
from scipy.optimize import fmin

from concepts.math.rotationlib_xyzw import quat_diff

__all__ = ['scipy_inverse_kinematics']


def _gen_ik_min_func(
    fk_func, pos, quat, lower_bound, upper_bound,
    additional_constraint_func=None,
    pos_weight=1.0, quat_weight=1.0, boundary_repulsion_weight=0.1, additional_loss_func=None, additional_loss_weight=1.0
):
    def func(qpos):
        if not np.all(lower_bound <= qpos) or not np.all(qpos <= upper_bound):
            return 1e-9
        if additional_constraint_func is not None:
            if not additional_constraint_func(qpos):
                return 1e-9

        fk_pos, fk_quat = fk_func(qpos)
        pos_error = np.linalg.norm(fk_pos - pos)
        quat_error = quat_diff(fk_quat, quat)

        rv = pos_error * pos_weight + quat_error * quat_weight

        lower_bound_distance = np.maximum(qpos - lower_bound, 0)
        upper_bound_distance = np.maximum(upper_bound - qpos, 0)

        if boundary_repulsion_weight > 0:
            # Only activate when the joint is close to the bound (within 0.1 rad).
            rv -= boundary_repulsion_weight * np.sum(np.minimum(lower_bound_distance, 0.1) ** 2)  # Maximize the distance to the lower bound.
            rv -= boundary_repulsion_weight * np.sum(np.minimum(upper_bound_distance, 0.1) ** 2)  # Maximize the distance to the upper bound.

        if additional_loss_func is not None:
            additional_loss = additional_loss_func(qpos)
            rv += additional_loss * additional_loss_weight

        return rv

    return func


def scipy_inverse_kinematics(
    fk_func, pos, quat, lower_bound, upper_bound,
    q0=None, sample_func=None,
    pos_weight=1.0, quat_weight=1.0, boundary_repulsion_weight=0.1,
    additional_constraint_func=None, additional_loss_func=None, additional_loss_weight=1.0,
    nr_trials=50, verbose=False
):
    if sample_func is None and q0 is None:
        raise ValueError('Either q0 or sample_func should be provided.')
    if sample_func is None and q0 is not None:
        sample_func = lambda: q0

    func = _gen_ik_min_func(
        fk_func, pos, quat,
        lower_bound, upper_bound,
        pos_weight=pos_weight, quat_weight=quat_weight,
        boundary_repulsion_weight=boundary_repulsion_weight,
        additional_constraint_func=additional_constraint_func,
        additional_loss_func=additional_loss_func,
        additional_loss_weight=additional_loss_weight
    )

    best_loss = np.inf
    best_qpos = None
    for i in range(nr_trials):
        init_guess = sample_func()
        qpos = fmin(func, x0=init_guess, disp=verbose)
        loss = func(qpos)
        if loss < best_loss:
            best_loss = loss
            best_qpos = qpos

    return best_qpos
