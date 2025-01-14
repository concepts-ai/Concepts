#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pivot_sampler.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/31/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Iterator, List, NamedTuple, Optional, Tuple

import numpy as np
import open3d as o3d

from concepts.dm.crowhat.manipulation_utils.contact_point_sampler import ContactPointProposal, gen_candidate_pose_b_from_two_contact_points, gen_robot_qpos_from_attached_object_pose, get_desired_pose_b_from_current_object_poses
from concepts.dm.crowhat.manipulation_utils.contact_point_sampler import pairwise_sample, gen_contact_point_with_normal_constraint
from concepts.dm.crowhat.world.manipulator_interface import SingleGroupMotionPlanningInterface

from concepts.dm.crowhat.world.planning_world_interface import PlanningWorldInterface
from concepts.math.rotationlib_xyzw import axisangle2quat, pos_quat2mat, quat_identity, quat_mul, rotate_vector, slerp

from concepts.utils.typing_utils import Vec3f, Vec4f

__all__ = [
    'RotationAlongAxis', 'gen_contact_point_for_indirect_rotation',
    'calc_object_rotation_pose_trajectory',
    'SingleArmIndirectPivotParameter', 'gen_single_arm_indirect_pivot_parameter', 'check_collision_for_two_arm_indirect_pivot_parameters',
    'calc_single_arm_indirect_pivot_qpos_trajectory', 'calc_single_arm_indirect_pivot_ee_pose_trajectory'
]


class RotationAlongAxis(NamedTuple):
    """The rotation information that specifies the rotation center, the rotation axis, and the rotation angle."""

    center: Vec3f
    axis: Vec3f
    angle: float


def gen_contact_point_for_indirect_rotation(planning_world: PlanningWorldInterface, object_id: int, rotation: RotationAlongAxis, nr_trials: int = 10000, batch_size: int = 100, max_returns: int = 1000, normal_tol: float = 0.1) -> np.ndarray:
    """Sample contact points on the object so that we can apply force on the object to rotate it along the given axis.

    Args:
        planning_world: the PlanningWorldInterface instance.
        object_id: the ID of the object to rotate.
        rotation: the rotation information.
        nr_trials: the number of trials to sample the contact points.
        batch_size: the number of points to sample in each trial.
        max_returns: the maximum number of contact points to return.
        normal_tol: the tolerance of the dot product between the normal and the rotation. If the dot product is bigger than this value (meaning that they are not perpendicular), the point will be discarded.
    """
    center, axis, angle = rotation
    mesh = planning_world.get_object_mesh(object_id)

    from concepts.simulator.pybullet.manipulation_utils.contact_samplers import _sample_points_uniformly

    nr_returns = 0
    for i in range(nr_trials // batch_size):
        sampled_pcd = _sample_points_uniformly(mesh, batch_size, use_triangle_normal=True)
        sampled_points= np.asarray(sampled_pcd.points)
        sampled_normals = np.asarray(sampled_pcd.normals)

        # Check if the normals are not pointing towards the rotation axis.
        dot_products = np.abs(np.dot(sampled_normals, axis))
        remaining_indices = np.where(dot_products < normal_tol)[0]

        for idx in remaining_indices:
            yield ContactPointProposal(object_id, sampled_points[idx], sampled_normals[idx])
            nr_returns += 1
            if nr_returns >= max_returns:
                return


def calc_object_rotation_pose_trajectory(planning_world: PlanningWorldInterface, object_id: int, support_id: int, rotation: RotationAlongAxis, nr_steps: int = 100, min_distance_from_support: float = 0.0) -> list:
    """Generate an object pose trajectory that rotates the object along the given axis by the given angle. This function asserts that the object is currently
    resting on a support surface with the given `support_id`.  During its rotation, the object will not penetrate the support surface.

    The rotation center and the axis specifies a ray in the 3D space, and the object will rotate around this ray. It can be generated, by, for example,
    fitting the principle axes of the object point cloud and using the center and one of the principle axes as the rotation center and axis.

    Args:
        planning_world: the PlanningWorldInterface instance.
        object_id: the ID of the object to rotate.
        support_id: the ID of the support surface.
        rotation: the rotation information.
        nr_steps: the number of steps in the trajectory.
    """

    assert nr_steps >= 1

    center, axis, angle = rotation
    pcd = planning_world.get_object_point_cloud(object_id)

    pos, quat = planning_world.get_object_pose(object_id)
    original_pose_matrix = pos_quat2mat(pos, quat)
    quat_target = axisangle2quat(axis, angle)

    support_center, support_normal = planning_world.get_single_contact_normal(object_id, support_id, return_center=True)
    support_normal = support_normal / np.linalg.norm(support_normal)

    center = np.array(center)

    trajectory = list()
    for i in range(nr_steps):
        t = i / (nr_steps - 1)
        angle = angle * t
        quat_new = slerp(quat_identity(), quat_target, t)

        step_pos = center + rotate_vector(pos - center, quat_new)
        step_quat = quat_mul(quat_new, quat)

        # Check if the object is penetrating the support surface.
        step_pose_matrix = pos_quat2mat(step_pos, step_quat)
        open3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
        step_open3d_pcd = open3d_pcd.transform(np.linalg.inv(original_pose_matrix)).transform(step_pose_matrix)
        step_pcd = np.asarray(step_open3d_pcd.points)

        # Compute the minimum distance between the object and the support surface.
        min_distance = np.min(np.dot(step_pcd - support_center, support_normal))

        if min_distance < min_distance_from_support:
            step_pos += (min_distance_from_support - min_distance) * support_normal

        trajectory.append((step_pos, step_quat))

    return trajectory


class SingleArmIndirectPivotParameter(NamedTuple):
    object_id: int
    tool_id: int
    support_id: int
    rotation: RotationAlongAxis
    contact_point_on_a: ContactPointProposal
    contact_point_on_b: ContactPointProposal
    b_pose: Tuple[Vec3f, Vec4f]
    pre_contact_distance: float
    post_contact_distance: float
    robot_qpos: np.ndarray
    robot_ee_pose: Tuple[Vec3f, Vec4f]


def gen_single_arm_indirect_pivot_parameter(
    planning_world: PlanningWorldInterface, robot: SingleGroupMotionPlanningInterface, object_id: int, tool_id: int, support_id: int, rotation: RotationAlongAxis,
    max_returns: int = 1, contact_point_max_returns: int = 100, tool_pose_nr_trials: int = 4,
    pre_contact_distance: float = 0.01, post_contact_distance: float = 0.005,
    verbose: bool = False
) -> Iterator[SingleArmIndirectPivotParameter]:
    nr_returns = 0
    for x, y in pairwise_sample(
        gen_contact_point_for_indirect_rotation(planning_world, object_id, rotation, max_returns=contact_point_max_returns),
        gen_contact_point_with_normal_constraint(planning_world, tool_id, normal_constraint=None, max_returns=contact_point_max_returns)
    ):
        if verbose:
            print('gen_single_arm_indirect_pivot_parameter::sampled_contact_points:', x, y)
        for tool_pose in gen_candidate_pose_b_from_two_contact_points(planning_world, object_id, tool_id, x, y, distance=pre_contact_distance, nr_trials=tool_pose_nr_trials):
            if verbose:
                print('gen_single_arm_indirect_pivot_parameter::sampled_tool_pose:', tool_pose)
            for ee_pose, qpos in gen_robot_qpos_from_attached_object_pose(planning_world, robot, tool_pose[0], tool_pose[1]):
                if verbose:
                    print('gen_single_arm_indirect_pivot_parameter::sampled_ee_pose:', ee_pose, 'qpos:', qpos)
                yield SingleArmIndirectPivotParameter(object_id, tool_id, support_id, rotation, x, y, tool_pose, pre_contact_distance, post_contact_distance, qpos, ee_pose)
                nr_returns += 1
                if nr_returns >= max_returns:
                    return


def check_collision_for_two_arm_indirect_pivot_parameters(planning_world: PlanningWorldInterface, robot1: SingleGroupMotionPlanningInterface, robot2: SingleGroupMotionPlanningInterface, pivot1: SingleArmIndirectPivotParameter, pivot2: SingleArmIndirectPivotParameter) -> bool:
    with planning_world.checkpoint_world():
        robot1.set_qpos(pivot1.robot_qpos)
        robot2.set_qpos(pivot2.robot_qpos)

        possible_collision_pairs = [
            (robot1.body_id, robot2.body_id),
            (pivot1.tool_id, pivot2.tool_id),
            (robot1.body_id, pivot2.tool_id),
            (robot2.body_id, pivot1.tool_id)
        ]

        for contact in planning_world.get_contact_points():
            if (contact.body_a, contact.body_b) in possible_collision_pairs or (contact.body_b, contact.body_a) in possible_collision_pairs:
                return True


def calc_single_arm_indirect_pivot_qpos_trajectory(
    planning_world: PlanningWorldInterface, robot: SingleGroupMotionPlanningInterface, pivot_parameter: SingleArmIndirectPivotParameter,
) -> Optional[List[np.ndarray]]:
    object_pose_trajectory = calc_object_rotation_pose_trajectory(planning_world, pivot_parameter.object_id, pivot_parameter.support_id, pivot_parameter.rotation, nr_steps=10, min_distance_from_support=0.01)
    tool_pose_trajectory = [(pivot_parameter.b_pose[0], pivot_parameter.b_pose[1])]
    ee_pose_trajectory = [pivot_parameter.robot_ee_pose]
    qpos_trajectory = [pivot_parameter.robot_qpos]

    if pivot_parameter.pre_contact_distance > 0:
        prev_qpos = qpos_trajectory[-1]
        prev_tool_pose = tool_pose_trajectory[-1]
        prev_ee_pose = ee_pose_trajectory[-1]

        distance_along_normal = pivot_parameter.pre_contact_distance + pivot_parameter.post_contact_distance
        desired_tool_pose = (prev_tool_pose[0] - distance_along_normal * pivot_parameter.contact_point_on_a.normal, pivot_parameter.b_pose[1])
        desired_ee_pose = get_desired_pose_b_from_current_object_poses(prev_tool_pose, prev_ee_pose, desired_tool_pose)
        qpos = robot.ik(desired_ee_pose[0], desired_ee_pose[1], qpos=prev_qpos, max_distance=0.5)

        if qpos is None:
            return None

        tool_pose_trajectory.append(desired_tool_pose)
        ee_pose_trajectory.append(desired_ee_pose)
        qpos_trajectory.append(qpos)

    for i, (prev_obj_pose, desired_obj_pose) in enumerate(zip(object_pose_trajectory, object_pose_trajectory[1:])):
        prev_qpos = qpos_trajectory[-1]
        prev_tool_pose = tool_pose_trajectory[-1]
        prev_ee_pose = ee_pose_trajectory[-1]

        desired_tool_pose = get_desired_pose_b_from_current_object_poses(prev_obj_pose, prev_tool_pose, desired_obj_pose)
        desired_ee_pose = get_desired_pose_b_from_current_object_poses(prev_tool_pose, prev_ee_pose, desired_tool_pose)

        qpos = robot.ik(desired_ee_pose[0], desired_ee_pose[1], qpos=prev_qpos, max_distance=0.5)

        if qpos is None:
            qpos_prime = robot.ik(desired_ee_pose[0], desired_ee_pose[1])
            if qpos_prime is not None:
                reason = 'The desired pose is too far from the current pose.'
            else:
                reason = 'The desired pose is unreachable.'
            print(f'Failed to find IK solution at step {i}. Reason: {reason}')
            return None

        tool_pose_trajectory.append(desired_tool_pose)
        ee_pose_trajectory.append(desired_ee_pose)
        qpos_trajectory.append(qpos)

    return qpos_trajectory


def calc_single_arm_indirect_pivot_ee_pose_trajectory(
    planning_world: PlanningWorldInterface, robot: SingleGroupMotionPlanningInterface,
    pivot_parameter: SingleArmIndirectPivotParameter,
) -> Optional[List[Tuple[Vec3f, Vec4f]]]:
    object_pose_trajectory = calc_object_rotation_pose_trajectory(planning_world, pivot_parameter.object_id, pivot_parameter.support_id, pivot_parameter.rotation, nr_steps=10, min_distance_from_support=0.01)
    tool_pose_trajectory = [(pivot_parameter.b_pose[0], pivot_parameter.b_pose[1])]
    ee_pose_trajectory = [pivot_parameter.robot_ee_pose]
    pose_trajectory = [pivot_parameter.robot_ee_pose]
    for i, (prev_obj_pose, desired_obj_pose) in enumerate(zip(object_pose_trajectory, object_pose_trajectory[1:])):
        prev_tool_pose = tool_pose_trajectory[-1]
        prev_ee_pose = ee_pose_trajectory[-1]

        desired_tool_pose = get_desired_pose_b_from_current_object_poses(prev_obj_pose, prev_tool_pose, desired_obj_pose)
        desired_ee_pose = get_desired_pose_b_from_current_object_poses(prev_tool_pose, prev_ee_pose, desired_tool_pose)

        tool_pose_trajectory.append(desired_tool_pose)
        ee_pose_trajectory.append(desired_ee_pose)
        pose_trajectory.append(desired_ee_pose)

    return pose_trajectory
