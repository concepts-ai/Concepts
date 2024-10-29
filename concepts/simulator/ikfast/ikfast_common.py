#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : ikfast_common.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/26/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import itertools
from typing import Optional, Iterable, Tuple, List, Callable

import random
import numpy as np
import numpy.random as npr

from jacinle.logging import get_logger

from concepts.math.rotationlib_wxyz import mat2quat, quat2mat, rotate_vector, quat_mul, quat_conjugate

logger = get_logger(__file__)


class IKFastWrapperBase(object):
    def __init__(
        self, module,
        joint_ids: List[int], free_joint_ids: List[int],
        joints_lower: np.ndarray, joints_upper: np.ndarray,
        use_xyzw: bool = True,  # PyBullet uses xyzw.
        max_attempts: int = 1000,
        fix_free_joint_positions: bool = False,
        shuffle_solutions: bool = False,
        sort_closest_solution: bool = False,
        current_joint_position_getter: Optional[Callable[[], np.ndarray]] = None,
    ):
        """IKFast wrapper base class.

        Args:
            module: the IKFast module.
            joint_ids: the joint ids of the robot.
            free_joint_ids: the free joint ids of the robot.
            joints_lower: the lower limits of the joints.
            joints_upper: the upper limits of the joints.
            use_xyzw: whether to use xyzw for quaternion representation.
            max_attempts: the maximum number of attempts for IK.
            fix_free_joint_positions: whether to fix the free joint positions.
            shuffle_solutions: whether to shuffle the solutions.
            sort_closest_solution: whether to sort the solutions by the closest one.
            current_joint_position_getter: the getter for the current joint positions.
        """
        self.module = module
        self.joint_ids = joint_ids
        self.free_joint_ids = free_joint_ids

        self.use_xyzw = use_xyzw
        self.max_attempts = max_attempts

        self.joints_lower = joints_lower
        self.joints_upper = joints_upper

        self.free_joints_lower = list()
        self.free_joints_upper = list()
        for i, joint_id in enumerate(self.joint_ids):
            if joint_id in free_joint_ids:
                self.free_joints_lower.append(joints_lower[i])
                self.free_joints_upper.append(joints_upper[i])
        self.free_joints_lower = np.array(self.free_joints_lower)
        self.free_joints_upper = np.array(self.free_joints_upper)

        self.current_joint_position_getter = current_joint_position_getter

        self.fix_free_joint_positions = fix_free_joint_positions
        if self.fix_free_joint_positions:
            self.initial_free_joint_positions = self.get_current_free_joint_positions()
        else:
            self.initial_free_joint_positions = None
        self.shuffle_solutions = shuffle_solutions
        self.sort_closest_solution = sort_closest_solution

    def get_current_joint_positions(self) -> np.ndarray:
        if self.current_joint_position_getter is not None:
            return self.current_joint_position_getter()
        raise NotImplementedError

    def get_current_free_joint_positions(self) -> np.ndarray:
        joints = self.get_current_joint_positions()
        return np.array([joints[i] for i, joint_id in enumerate(self.joint_ids) if joint_id in self.free_joint_ids])

    def fk(self, qpos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pos, mat = self.module.get_fk(list(qpos))
        quat = mat2quat(mat)
        if self.use_xyzw:
            return pos, quat[[1, 2, 3, 0]]
        return pos, quat

    def ik_internal(self, pos: np.ndarray, quat: np.ndarray, sampled: Optional[np.ndarray] = None, body_pos: Optional[np.ndarray] = None, body_quat: Optional[np.ndarray] = None) -> List[np.ndarray]:
        if self.use_xyzw:
            quat = quat[[3, 0, 1, 2]]
            if body_quat is not None:
                body_quat = body_quat[[3, 0, 1, 2]]
            else:
                body_quat = np.array([1, 0, 0, 0])
        else:
            if body_quat is None:
                body_quat = np.array([1, 0, 0, 0])

        if body_pos is None:
            body_pos = np.array([0, 0, 0])

        # print('ik_internal: pos', pos, 'quat', quat, 'sampled', sampled, 'body_pos', body_pos, 'body_quat', body_quat)

        # Transform the target pose to the local frame of the IKFast model.
        pos = rotate_vector(pos - body_pos, quat_conjugate(body_quat))
        quat = quat_mul(quat_conjugate(body_quat), quat)

        mat = quat2mat(quat)
        if sampled is None:
            solutions = self.module.get_ik(mat.tolist(), pos.tolist())
        else:
            solutions = self.module.get_ik(mat.tolist(), pos.tolist(), list(sampled))

        if solutions is None:
            return list()
        return [np.array(solution) for solution in solutions]

    def gen_ik(
        self, pos: np.ndarray, quat: np.ndarray,
        last_qpos: Optional[np.ndarray], max_attempts: Optional[int] = None, max_distance: float = float('inf'),
        body_pos: Optional[np.ndarray] = None, body_quat: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> Iterable[np.ndarray]:
        if last_qpos is None:
            current_joint_positions = self.get_current_joint_positions()
            current_free_joint_positions = self.get_current_free_joint_positions()
        else:
            current_joint_positions = last_qpos
            current_free_joint_positions = [last_qpos[i] for i, joint_id in enumerate(self.joint_ids) if joint_id in self.free_joint_ids]

        if self.fix_free_joint_positions:
            generator = [self.initial_free_joint_positions]
        else:
            generator = itertools.chain([current_free_joint_positions], gen_uniform_sample_joints(self.free_joints_lower, self.free_joints_upper))
            if max_attempts is not None:
                generator = itertools.islice(generator, max_attempts)
            else:
                generator = itertools.islice(generator, self.max_attempts)

        pos = np.array(pos)
        quat = np.array(quat)

        succeeded = False
        for sampled in generator:
            solutions = self.ik_internal(pos, quat, sampled, body_pos, body_quat)
            if self.shuffle_solutions:
                random.shuffle(solutions)
            sorted_solutions = list()
            for solution in solutions:
                # print('Checking solution: ', solution, 'lower', self.joints_lower, 'upper', self.joints_upper, check_joint_limits(solution, self.joints_lower, self.joints_upper))
                if check_joint_limits(solution, self.joints_lower, self.joints_upper):
                    if distance_fn(solution, current_joint_positions) < max_distance:
                        succeeded = True

                        # fk_pos, fk_quat = self.fk(solution.tolist())
                        from concepts.math.rotationlib_xyzw import quat_mul as quat_mul_wxyz, quat_conjugate as quat_conjugate_wxyz
                        # print('query (inside): ', pos, quat, 'solution: ', solution, 'fk', fk_pos, fk_quat, 'fk_diff', np.linalg.norm(fk_pos - pos), quat_mul_xyzw(quat_conjugate_xyzw(fk_quat), quat)[3])

                        sorted_solutions.append(solution)
                    elif verbose:
                        print(f'IK solution is too far from current joint positions: {solution} vs {current_joint_positions}')

            if self.sort_closest_solution:
                sorted_solutions.sort(key=lambda qpos: distance_fn(qpos, current_joint_positions))
            yield from sorted_solutions

        if not succeeded and max_attempts is None:
            logger.warning(f'Failed to find IK solution for {pos} {quat} after {self.max_attempts} attempts.')


def check_joint_limits(qpos: np.ndarray, lower_limits: np.ndarray, upper_limits: np.ndarray) -> bool:
    return np.all(np.logical_and(qpos >= lower_limits, qpos <= upper_limits))


def uniform_sample_joints(lower_limits: np.ndarray, upper_limits: np.ndarray) -> np.ndarray:
    return np.array([npr.uniform(lower, upper) for lower, upper in zip(lower_limits, upper_limits)])


def gen_uniform_sample_joints(lower_limits: np.ndarray, upper_limits: np.ndarray) -> Iterable[np.ndarray]:
    while True:
        yield uniform_sample_joints(lower_limits, upper_limits)


def random_select_solution(solutions: List[np.ndarray]) -> np.ndarray:
    return random.choice(solutions)


def distance_fn(qpos1: np.ndarray, qpos2: np.ndarray) -> float:
    return np.linalg.norm(np.array(qpos1) - np.array(qpos2), ord=2)


def closest_select_solution(solutions: List[np.ndarray], current_qpos: np.ndarray) -> np.ndarray:
    return min(solutions, key=lambda qpos: distance_fn(qpos, current_qpos))

