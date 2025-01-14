#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : operation_space_trajectory.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/23/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
from typing import Sequence, Tuple, NamedTuple
from concepts.utils.typing_utils import VecNf, Vec3f, Vec4f
from concepts.math.interpolation_utils import PoseSpline
from concepts.dm.crowhat.world.manipulator_interface import SingleGroupMotionPlanningInterface, MotionPlanningResult
from concepts.dm.crowhat.manipulation_utils.pose_utils import pose_distance


class OperationSpaceTrajectory(NamedTuple):
    qpos_trajectory: Sequence[VecNf]
    pose_trajectory: Sequence[Tuple[Vec3f, Vec4f]]


def gen_joint_trajectory_from_cartesian_path_with_differential_ik(
    arm: SingleGroupMotionPlanningInterface,
    start_qpos: VecNf,
    cartesian_path: Sequence[Tuple[Vec3f, Vec4f]],
    step_size: float = 0.01,
    target_tolerance: float = 0.01
):
    """Generate a joint trajectory from a given start qpos and a cartesian path using differential IK.

    This function starts with the given q0, and use the differential IK to iteratively generate the joint trajectory
    that follows the given cartesian path. This function is useful if we want our controller is an operation-space controller,
    possibly with some compliance control, but we want to verify the feasibility of the generated joint trajectory (e.g., joint limits and collision).

    Args:
        arm: the arm interface.
        start_qpos: the start qpos.
        cartesian_path: the cartesian path, a sequence of (position, quaternion) tuples.
        step_size: the step size for the differential IK.
        target_tolerance: the tolerance for the target pose. Once the current pose is within this tolerance to the last pose in the cartesian path, the motion is considered as solved.

    Returns:
        a MotionPlanningResult object, which wraps around the :class:`OperationSpaceTrajectory` object.
        If the motion planning is unsuccessful, the result will be a failure with the corresponding error message.
    """
    qpos = np.asarray(start_qpos)
    pose_spline = PoseSpline.from_pose_sequence(cartesian_path)
    goal_pose = cartesian_path[-1]

    qpos_trajectory = [qpos]
    pose_trajectory = [cartesian_path[0]]

    assert pose_distance(pose_trajectory[0], arm.fk(qpos)) < 1e-3, f'The start qpos does not match the start pose: {pose_trajectory[0]} vs {arm.fk(qpos)}'

    solved = False
    for i in range(1000):
        last_qpos, last_pose = qpos_trajectory[-1], pose_trajectory[-1]

        if pose_distance(last_pose, goal_pose) < target_tolerance:
            solved = True
            break

        _, next_pose_target = pose_spline.get_next(last_pose, step_size=1)
        dq = arm.calc_differential_ik_qpos_diff(last_qpos, last_pose, next_pose_target)
        next_qpos = last_qpos + dq / np.linalg.norm(dq) * step_size

        qpos_trajectory.append(next_qpos)
        pose_trajectory.append(arm.fk(next_qpos))

        progress = pose_distance(last_pose, pose_trajectory[-1])
        if progress < 1e-6:
            return MotionPlanningResult.fail(f'No progress made at step {i}. Current progress: {progress}.')

    if not solved:
        return MotionPlanningResult.fail('Failed to reach the goal pose after 1000 steps.')
    return MotionPlanningResult.success(OperationSpaceTrajectory(qpos_trajectory, pose_trajectory))

