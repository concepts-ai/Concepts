#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : planar_push_sampler.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/01/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import copy
import itertools
import tabulate
from dataclasses import dataclass
from typing import Optional, Iterator, Sequence, Tuple, List, Callable

import numpy as np
import open3d as o3d

import jacinle

from concepts.math.rotationlib_xyzw import find_orthogonal_vector, quat_mul, rotate_vector, enumerate_quaternion_from_vectors, quaternion_from_axes
from concepts.math.cad.mesh_utils import mesh_line_intersect
from concepts.dm.crowhat.world.planning_world_interface import PlanningWorldInterface
from concepts.dm.crowhat.world.manipulator_interface import SingleGroupMotionPlanningInterface
from concepts.dm.crowhat.manipulation_utils.contact_point_sampler import ContactPointProposal, _sample_points_uniformly


def gen_planar_push_contact_point(
    planning_world: PlanningWorldInterface, object_id: int, support_id: int, *,
    max_attempts: int = 1000, batch_size: int = 100,
    contact_normal_tol: float = 0.02,
    filter_push_dir: Optional[np.ndarray] = None,
    np_random: Optional[np.random.RandomState] = None,
    verbose: bool = False
) -> Iterator[ContactPointProposal]:
    if np_random is None:
        np_random = np.random

    mesh = planning_world.get_object_mesh(object_id)

    nr_batches = int(max_attempts / batch_size)
    feasible_point_indices = list()
    for _ in range(nr_batches):
        pcd = _sample_points_uniformly(mesh, batch_size, use_triangle_normal=True, seed=np_random.randint(0, 1000000))

        # get the contact points between the object and the support object
        # A special hack here: when the object is the support object, we assume the contact normal is [0, 0, 1].
        if object_id == support_id:
            contact_normal = np.array([0, 0, 1], dtype=np.float32)
        else:
            contact_normal = planning_world.get_single_contact_normal(object_id, support_id)

        # filter out the points that are not on the contact plane
        feasible_point_cond = np.abs(np.asarray(pcd.normals).dot(contact_normal)) < contact_normal_tol
        if filter_push_dir is not None:
            feasible_point_cond = np.logical_and(
                feasible_point_cond,
                np.asarray(pcd.normals, dtype=np.float32).dot(-filter_push_dir) > 0.8
            )

        feasible_point_indices = np.where(feasible_point_cond)[0]

        if len(feasible_point_indices) == 0:
            continue

        if verbose:
            rows = list()
            for index in feasible_point_indices:
                rows.append((index, pcd.points[index], -pcd.normals[index]))
            jacinle.lf_indent_print(tabulate.tabulate(rows, headers=['index', 'point', 'normal']))

        # create a new point cloud
        for index in feasible_point_indices:
            if verbose:
                jacinle.lf_indent_print('sample_push_with_support', 'point', pcd.points[index], 'normal', -pcd.normals[index])
            yield ContactPointProposal(object_id, np.asarray(pcd.points[index]), np.asarray(pcd.normals[index]))

    if len(feasible_point_indices) == 0:
        raise ValueError(f'No feasible points for {object_id} on {support_id} after {nr_batches * batch_size} attempts.')


@dataclass
class PlanarPushParameter(object):
    object_id: int
    push_pos: np.ndarray
    push_dir: np.ndarray
    prepush_distance: float
    push_distance: float

    robot_ee_pose: Tuple[np.ndarray, np.ndarray]

    @property
    def total_push_distance(self):
        return self.prepush_distance + self.push_distance


def gen_planar_push_parameter(
    planning_world: PlanningWorldInterface, robot: SingleGroupMotionPlanningInterface, object_id: int, support_id: int, *,
    prepush_distance: float = 0.05,
    push_distance_fn: Optional[Callable] = None,
    max_attempts: int = 1000, batch_size: int = 100,
    np_random: Optional[np.random.RandomState] = None,
    verbose: bool = False
) -> Iterator[PlanarPushParameter]:
    if push_distance_fn is None:
        push_distance_fn = lambda: 0.1

    for contact_point in gen_planar_push_contact_point(planning_world, object_id, support_id, max_attempts=max_attempts, batch_size=batch_size, np_random=np_random, verbose=verbose):
        yield PlanarPushParameter(
            contact_point.object_id, contact_point.point, -contact_point.normal, prepush_distance, push_distance_fn(),
            (contact_point.point, robot.ee_default_quat)
        )


def calc_push_ee_pose_trajectory(planar_push_parameter: PlanarPushParameter) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Given the name of the object, the push pose, the push direction, and the push distance, generate a trajectory of 6D poses that the robot should follow to push the object."""

    push_pos, push_dir = planar_push_parameter.push_pos, planar_push_parameter.push_dir
    prepush_distance, push_distance = planar_push_parameter.prepush_distance, planar_push_parameter.push_distance
    _, quat = planar_push_parameter.robot_ee_pose

    push_pos += (0, 0, 0.010)
    real_start_pos = push_pos - prepush_distance * push_dir / np.linalg.norm(push_dir)
    real_end_pos = push_pos + push_dir * push_distance

    trajectory = list()
    for i in range(10 + 1):
        pos = real_start_pos + (real_end_pos - real_start_pos) * i / 10
        trajectory.append((pos, quat))

    return trajectory


@dataclass
class PlanarIndirectPushParameter(object):
    object_id: int
    tool_id: int
    support_id: int

    push_pos: np.ndarray
    push_dir: np.ndarray

    contact_on_point_object: np.ndarray
    contact_on_point_tool: np.ndarray
    tool_pose: Tuple[np.ndarray, np.ndarray]

    prepush_distance: float
    push_distance: float

    @property
    def total_push_distance(self):
        return self.prepush_distance + self.push_distance


def gen_planar_indirect_push_parameter(
    planning_world: PlanningWorldInterface, target_id: int, tool_id: int, support_id: int, *,
    prepush_distance: float = 0.05,
    max_attempts: int = 10000, batch_size: int = 100, filter_push_dir: Optional[np.ndarray] = None,
    push_distance_distribution: Sequence[float] = (0.1, 0.15), push_distance_sample: bool = False,
    contact_normal_tol: float = 0.01,
    np_random: Optional[np.random.RandomState] = None,
    verbose: bool = False
) -> Iterator[PlanarIndirectPushParameter]:
    """Sample a push of the target object using the tool object. The target object should be placed on the support object.

    Args:
        planning_world: the planning world interface.
        target_id: the PyBullet body id of the target object.
        tool_id: the PyBullet body id of the tool object.
        support_id: the PyBullet body id of the support object.
        prepush_distance: the distance to pre-push the tool object before the actual push.
        max_attempts: the maximum number of attempts for sampling.
        batch_size: the number of samples in batch processing.
        filter_push_dir: if specified, the push direction will be filtered to be within 0.2 cosine distance from this direction.
        push_distance_distribution: the distribution of the push distance. If `push_distance_sample` is True, this will be used to sample the push distance.
        push_distance_sample: whether to sample the push distance from the distribution. If False, the function will enumerate the push distances in the distribution.
        contact_normal_tol: the tolerance for the contact normal. We will sample the push direction such that the contact normal is within `contact_normal_tol` degree from the gravity direction.
        np_random: the random state.
        verbose: whether to print the sampling information.
    """
    if np_random is None:
        np_random = np.random

    target_mesh = planning_world.get_object_mesh(target_id)
    tool_mesh = planning_world.get_object_mesh(tool_id)

    nr_batches = int(max_attempts / batch_size)
    if target_id == support_id:
        contact_normal = np.array([0, 0, 1], dtype=np.float32)
    else:
        contact_normal = planning_world.get_single_contact_normal(target_id, support_id)

    for _ in range(nr_batches):
        target_pcd = _sample_points_uniformly(target_mesh, batch_size, use_triangle_normal=True, seed=np_random.randint(0, 1000000))
        tool_pcd = _sample_points_uniformly(tool_mesh, batch_size, use_triangle_normal=True, seed=np_random.randint(0, 1000000))

        # feasible_target_point_cond = np.abs(np.asarray(target_pcd.normals).dot(contact_normal)) < 0.01 # 0.1 for real demo.
        feasible_target_point_cond = np.abs(np.asarray(target_pcd.normals).dot(contact_normal)) < contact_normal_tol

        if filter_push_dir is not None:
            feasible_target_point_cond = np.logical_and(
                feasible_target_point_cond,
                np.asarray(target_pcd.normals, dtype=np.float32).dot(-filter_push_dir) > 0.8
            )

        feasible_target_point_indices = np.where(feasible_target_point_cond)[0]

        all_index_pairs = list(itertools.product(feasible_target_point_indices, range(batch_size)))
        np_random.shuffle(all_index_pairs)
        for target_index, tool_index in all_index_pairs:
            target_point_pos = np.asarray(target_pcd.points[target_index])
            target_point_normal = -np.asarray(target_pcd.normals[target_index])  # point inside

            tool_point_pos = np.asarray(tool_pcd.points[tool_index])
            tool_point_normal = np.asarray(tool_pcd.normals[tool_index])  # point outside (towards the tool)

            current_tool_pos, current_tool_quat = planning_world.get_object_pose(tool_id)

            # Solve for a quaternion that aligns the tool normal with the target normal
            for rotation_quat in enumerate_quaternion_from_vectors(tool_point_normal, target_point_normal, 4):
                # This is the world coordinate for the tool point after rotation.
                new_tool_point_pos = current_tool_pos + rotate_vector(tool_point_pos - current_tool_pos, rotation_quat)
                # Now compute the displacement for the tool object
                final_tool_pos = target_point_pos - new_tool_point_pos + current_tool_pos
                final_tool_pos -= target_point_normal * prepush_distance
                final_tool_quat = quat_mul(rotation_quat, current_tool_quat)

                success = True
                with planning_world.save_world():
                    planning_world.set_object_pose(tool_id, (final_tool_pos, final_tool_quat))
                    if planning_world.check_collision_with_other_objects(tool_id):
                        success = False

                if success:
                    if push_distance_sample:
                        distances = [np_random.choice(push_distance_distribution)]
                    else:
                        distances = push_distance_distribution

                    for distance in distances:
                        yield PlanarIndirectPushParameter(
                            target_id, tool_id, support_id,
                            push_pos=target_point_pos, push_dir=target_point_normal,
                            contact_on_point_object=target_point_pos, contact_on_point_tool=tool_point_pos,
                            tool_pose=(final_tool_pos, final_tool_quat),
                            prepush_distance=prepush_distance, push_distance=distance
                        )
