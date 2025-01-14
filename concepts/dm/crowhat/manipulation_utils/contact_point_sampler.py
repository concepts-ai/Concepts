#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : contact_point_sampler.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/31/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Basic samplers for contact points and contact pairs."""

import itertools
from typing import Iterator, NamedTuple, Optional, Tuple

import numpy as np
import open3d as o3d

from concepts.dm.crowhat.world.manipulator_interface import SingleGroupMotionPlanningInterface
from concepts.dm.crowhat.world.planning_world_interface import PlanningWorldInterface
from concepts.math.rotationlib_xyzw import enumerate_quaternion_from_vectors, quat_mul, quat_conjugate, rotate_vector
from concepts.math.frame_utils_xyzw import solve_ee_from_tool, get_transform_a_to_b
from concepts.utils.typing_utils import Vec3f, Vec4f, Open3DPointCloud, Open3DTriangleMesh

__all__ = [
    'ContactPointProposal', 'gen_contact_point_with_normal_constraint', 'pairwise_sample',
    'gen_candidate_pose_b_from_two_contact_points', 'gen_ee_pose_from_contact_point',
    'gen_robot_qpos_from_attached_object_pose',
    'get_desired_pose_b_from_current_object_poses',
    '_sample_points_uniformly'
]


class ContactPointProposal(NamedTuple):
    """A proposal of a contact point on an object."""

    object_id: int
    """The identifier of the object."""

    point: Vec3f
    """The position of the contact point."""

    normal: Vec3f
    """The normal of the contact point, pointing outwards."""


def gen_contact_point_with_normal_constraint(planning_world: PlanningWorldInterface, object_id: int, normal_constraint: Optional[Vec3f], normal_tol: float = 0.1, nr_trials: int = 10000, batch_size: int = 100, max_returns: int = 1000) -> np.ndarray:
    """Sample contact points on the object with the given normal constraint.

    Args:
        planning_world: the PlanningWorldInterface instance.
        object_id: the ID of the object to sample the contact points.
        normal_constraint: the normal constraint.
        normal_tol: the tolerance of the dot product between the normal and the constraint. If the dot product is bigger than this value (meaning that they are not perpendicular), the point will be discarded.
        nr_trials: the number of trials to sample the contact points.
        batch_size: the number of points to sample in each trial.
        max_returns: the maximum number of contact points to return.
    """
    mesh = planning_world.get_object_mesh(object_id)

    nr_returns = 0
    for i in range(nr_trials // batch_size):
        sampled_pcd = _sample_points_uniformly(mesh, batch_size, use_triangle_normal=True)
        sampled_points= np.asarray(sampled_pcd.points)
        sampled_normals = np.asarray(sampled_pcd.normals)

        if normal_constraint is None:
            remaining_indices = range(len(sampled_normals))
        else:
            # Check if the normals are not pointing towards the rotation axis.
            dot_products = np.dot(sampled_normals, normal_constraint)
            remaining_indices = np.where(dot_products > 1 - normal_tol)[0]

        for idx in remaining_indices:
            yield ContactPointProposal(object_id, sampled_points[idx], sampled_normals[idx])
            nr_returns += 1
            if nr_returns >= max_returns:
                return


def pairwise_sample(sampler1, sampler2, product: bool = False, max_returns: int = 1000):
    """Pairwise sample from two samplers.

    Args:
        sampler1: the first sampler.
        sampler2: the second sampler.
        product: whether to return the product of the two samplers. If False, the two samplers will be zipped.
        max_returns: the maximum number of pairs to return.

    Returns:
        an iterator of samples from the two samplers.
    """

    if product:
        return itertools.islice(itertools.product(sampler1, sampler2), max_returns)
    return itertools.islice(zip(sampler1, sampler2), max_returns)


def gen_candidate_pose_b_from_two_contact_points(planning_world: PlanningWorldInterface, a: int, b: int, contact_point_a: ContactPointProposal, contact_point_b: ContactPointProposal, nr_trials: int = 4, distance: float = 0.01) -> Iterator[Tuple[Vec3f, Vec4f]]:
    """Generate candidate poses of object B by moving object B so that the `contact_point_b` aligns with the `contact_point_a`. This function assumes that we are going to align
    `contact_point_b.normal` with `-contact_point_a.normal`. That is, B will exert a force on A "inwards".

    Args:
        planning_world: the PlanningWorldInterface instance.
        a: the ID of object A.
        b: the ID of object B.
        contact_point_a: the contact point on object A.
        contact_point_b: the contact point on object B.
        nr_trials: the number of trials to sample the contact points.
        distance: the distance between the contact points.

    Returns:
        an iterator of the possible poses of object B.
    """

    pos1 = contact_point_a.point
    normal1 = -contact_point_a.normal
    pos2 = contact_point_b.point
    normal2 = contact_point_b.normal

    current_b_pos, current_b_quat = planning_world.get_object_pose(b)

    for rotation_quat in enumerate_quaternion_from_vectors(normal2, normal1, nr_trials):
        new_b_point_pos = current_b_pos + rotate_vector(pos2 - current_b_pos, rotation_quat)
        final_b_pos = pos1 - new_b_point_pos + current_b_pos - normal1 * distance
        final_b_quat = quat_mul(rotation_quat, current_b_quat)

        # Check the collision between two objects.
        with planning_world.checkpoint_world():
            planning_world.set_object_pose(b, (final_b_pos, final_b_quat))
            if planning_world.check_collision(a, b):
                continue
        yield final_b_pos, final_b_quat


def gen_ee_pose_from_contact_point(planning_world: PlanningWorldInterface, robot: SingleGroupMotionPlanningInterface, object_id: int, contact_point: ContactPointProposal, distance: float = 0.01, z_delta: float = 0.0, nr_trials: int = 4, side: str = 'front') -> Iterator[Tuple[Vec3f, Vec4f]]:
    """Generate the end-effector pose in order to push the object with the given contact point.

    Args:
        planning_world: the PlanningWorldInterface instance.
        robot: the robot instance.
        object_id: the ID of the object.
        contact_point: the contact point.
        distance: the distance between the contact point and the end-effector.
        nr_trials: the number of trials to sample the end-effector poses.
        side: the side of the hand to push. It can be either 'front' or 'back' or 'down'.
    """

    pos, normal = contact_point.point, contact_point.normal
    pos = pos + normal * distance
    if z_delta != 0:
        pos = pos + np.array([0, 0, z_delta], dtype=np.float32)

    if side == 'front':
        ee_contact_normal = np.array([1, 0, 0], dtype=np.float32)
    elif side == 'down':
        ee_contact_normal = np.array([0, 0, -1], dtype=np.float32)
    elif side == 'back':
        ee_contact_normal = np.array([-1, 0, 0], dtype=np.float32)
    else:
        raise ValueError(f"Invalid hand contact side: {side}")

    for rotation_quat in enumerate_quaternion_from_vectors(ee_contact_normal, -normal, nr_trials):
        quat = quat_mul(rotation_quat, robot.ee_default_quat)
        yield pos, quat


def gen_robot_qpos_from_attached_object_pose(planning_world: PlanningWorldInterface, robot: SingleGroupMotionPlanningInterface, pos: Vec3f, quat: Vec4f, nr_trials: int = 10, max_returns: int = 1) -> Iterator[Tuple[Tuple[Vec3f, Vec4f], np.ndarray]]:
    """Generate the robot qpos with the attached object pose.

    Args:
        planning_world: the PlanningWorldInterface instance.
        robot: the robot instance.
        pos: the position of the object.
        quat: the quaternion of the object.

    Returns:
        an iterator of possible robot qpos information, as tuples of end-effector pose and qpos.
    """

    ee_pos, ee_quat = robot.calc_ee_pose_from_single_attached_object_pose(pos, quat)
    nr_returns = 0
    for i in range(nr_trials):
        qpos = robot.ik(ee_pos, ee_quat)
        if qpos is not None:
            if robot.check_collision(qpos):
                continue
            yield (ee_pos, ee_quat), qpos
            nr_returns += 1
            if nr_returns >= max_returns:
                return


def get_desired_pose_b_from_current_object_poses(current_a_pose: Tuple[Vec3f, Vec4f], current_b_pose: Tuple[Vec3f, Vec4f], desired_a_pose: Tuple[Vec3f, Vec4f]):
    """Compute the desired pose of object B so that it can push object A to the desired pose. It assumes that the relative pose between A and B is fixed.

    Args:
        current_a_pose: the current pose of object A.
        current_b_pose: the current pose of object B.
        desired_a_pose: the desired pose of object A.
    """
    current_a_pos, current_a_quat = current_a_pose
    current_b_pos, current_b_quat = current_b_pose
    desired_a_pos, desired_a_quat = desired_a_pose

    return solve_ee_from_tool(desired_a_pos, desired_a_quat, get_transform_a_to_b(current_b_pos, current_b_quat, current_a_pos, current_a_quat))


def _sample_points_uniformly(pcd: Open3DTriangleMesh, nr_points: int, use_triangle_normal: bool = False, seed: Optional[int] = None) -> Open3DPointCloud:
    if seed is not None:
        o3d.utility.random.seed(seed)
    return pcd.sample_points_uniformly(nr_points, use_triangle_normal=use_triangle_normal)
