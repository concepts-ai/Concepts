#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pick_place_sampler.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/31/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import itertools
from typing import Optional, Iterator, Tuple, List, NamedTuple
from jacinle.utils.cache import cached_property

import numpy as np
import open3d as o3d

from concepts.math.rotationlib_xyzw import find_orthogonal_vector, quat_mul, rotate_vector, enumerate_quaternion_from_vectors, quaternion_from_axes
from concepts.math.cad.mesh_utils import mesh_line_intersect
from concepts.dm.crowhat.world.planning_world_interface import PlanningWorldInterface
from concepts.dm.crowhat.world.manipulator_interface import SingleArmMotionPlanningInterface
from concepts.dm.crowhat.manipulation_utils.contact_point_sampler import ContactPointProposal, gen_ee_pose_from_contact_point, pairwise_sample, _sample_points_uniformly
from concepts.dm.crowhat.manipulation_utils.path_generation_utils import is_collision_free_qpos

__all__ = [
    'GraspContactPairProposal', 'gen_grasp_contact_pair',
    'GraspParameter', 'gen_grasp_parameter',
    'calc_grasp_approach_ee_pose_trajectory',
    'PlacementParameter', 'gen_placement_parameter'
]


class GraspContactPairProposal(NamedTuple):
    object_id: int
    point1: np.ndarray
    normal1: np.ndarray
    point2: np.ndarray
    normal2: np.ndarray

    center: np.ndarray
    distance: float

    @cached_property
    def contact_point1(self) -> ContactPointProposal:
        return ContactPointProposal(self.object_id, self.point1, self.normal1)

    @cached_property
    def contact_point2(self) -> ContactPointProposal:
        return ContactPointProposal(self.object_id, self.point2, self.normal2)

    @classmethod
    def from_contact_points(cls, contact_point1: ContactPointProposal, contact_point2: ContactPointProposal):
        return cls(
            object_id=contact_point1.object_id,
            point1=contact_point1.point, normal1=contact_point1.normal,
            point2=contact_point2.point, normal2=contact_point2.normal,
            center=(contact_point1.point + contact_point2.point) / 2,
            distance=np.linalg.norm(contact_point1.point - contact_point2.point)
        )


def gen_grasp_contact_pair(
    planning_world: PlanningWorldInterface,
    object_id: int,
    gripper_distance: float,
    *,
    gripper_min_distance: float = 0.0001,
    surface_pointing_tol: float = 0.9,
    mesh_filename: Optional[str] = None, mesh_scale: float = 1.0,
    max_test_points_before_first: int = 250, max_test_points: int = 100000000, batch_size: int = 100,
    max_intersection_distance: float = 10,
    np_random: Optional[np.random.RandomState] = None,
    verbose: bool = False
) -> Iterator[GraspContactPairProposal]:
    """Given the name of the object, sample a 6D grasp pose. Before calling this function, the program should make sure that the gripper is open."""

    if np_random is None:
        np_random = np.random

    mesh = planning_world.get_object_mesh(object_id, mesh_filename=mesh_filename, mesh_scale=mesh_scale)
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    found = False
    nr_test_points_before_first = 0

    for _ in range(int(max_test_points / batch_size)):
        # TODO(Jiayuan Mao @ 2023/01/02): accelerate the computation.
        pcd = _sample_points_uniformly(mesh, batch_size, use_triangle_normal=True, seed=np_random.randint(0, 1000000))

        indices = list(range(len(pcd.points)))
        np_random.shuffle(indices)
        for i in indices:
            if not found:
                nr_test_points_before_first += 1

            point = np.asarray(pcd.points[i])
            normal = np.asarray(pcd.normals[i])

            if verbose:
                print('sample_grasp_v2_gen', 'point', point, 'normal', normal)

            point2 = point - normal * max_intersection_distance
            other_intersection = mesh_line_intersect(t_mesh, point2, normal)

            if verbose:
                print('  other_intersection', other_intersection)

            # if no intersection, try the next point.
            if other_intersection is None:
                if verbose:
                    print('  skip: no intersection')
                continue

            other_point, other_normal = other_intersection

            # if two intersection points are too close, try the next point.
            if np.linalg.norm(other_point - point) < gripper_min_distance:
                if verbose:
                    print('  skip: too close')
                continue

            # if the surface normals are too different, try the next point.
            if np.abs(np.dot(normal, other_normal)) < surface_pointing_tol:
                if verbose:
                    print('  skip: normal too different')
                continue

            grasp_center = (point + other_point) / 2
            grasp_distance = np.linalg.norm(point - other_point)

            if grasp_distance > gripper_distance:
                if verbose:
                    print('  skip: too far')
                continue

            found = True
            yield GraspContactPairProposal(
                object_id=object_id,
                point1=point, normal1=normal,
                point2=other_point, normal2=other_normal,
                center=grasp_center, distance=grasp_distance
            )

        if not found and nr_test_points_before_first > max_test_points_before_first:
            if verbose:
                print(f'Failed to find a grasp after {nr_test_points_before_first} points tested.')
            return


def gen_grasp_contact_pair_with_support_constraint(
    planning_world: PlanningWorldInterface,
    object_id: int, support_id: int,
    gripper_distance: float,
    *,
    gripper_min_distance: float = 0.0001,
    surface_pointing_tol: float = 0.9,
    support_normal_tol: float = 0.1,
    mesh_filename: Optional[str] = None, mesh_scale: float = 1.0,
    max_test_points_before_first: int = 250, max_test_points: int = 100000000, batch_size: int = 100,
    max_intersection_distance: float = 10,
    np_random: Optional[np.random.RandomState] = None,
    verbose: bool = False
) -> Iterator[GraspContactPairProposal]:
    support_normal = planning_world.get_single_contact_normal(object_id, support_id)
    for grasp_contact_pair in gen_grasp_contact_pair(
        planning_world, object_id, gripper_distance,
        gripper_min_distance=gripper_min_distance,
        surface_pointing_tol=surface_pointing_tol,
        mesh_filename=mesh_filename, mesh_scale=mesh_scale,
        max_test_points_before_first=max_test_points_before_first, max_test_points=max_test_points, batch_size=batch_size,
        max_intersection_distance=max_intersection_distance,
        np_random=np_random, verbose=verbose
    ):
        if np.abs(np.dot(support_normal, grasp_contact_pair.normal1)) < support_normal_tol:
            yield grasp_contact_pair


class GraspParameter(NamedTuple):
    contact_pair: GraspContactPairProposal
    robot_ee_pose: Tuple[np.ndarray, np.ndarray]
    robot_qpos: np.ndarray


def gen_grasp_parameter(
    planning_world: PlanningWorldInterface, robot: SingleArmMotionPlanningInterface,
    object_id: int, gripper_distance: float, *,
    gripper_min_distance: float = 0.0001,
    surface_pointing_tol: float = 0.9,
    mesh_filename: Optional[str] = None, mesh_scale: float = 1.0,
    max_test_points_before_first: int = 250, max_test_points: int = 100000000, batch_size: int = 100,
    max_intersection_distance: float = 10,
    np_random: Optional[np.random.RandomState] = None,
    verbose: bool = False
) -> Iterator[GraspParameter]:
    for grasp_contact_pair in gen_grasp_contact_pair(
        planning_world, object_id, gripper_distance,
        max_intersection_distance=max_intersection_distance,
        mesh_filename=mesh_filename, mesh_scale=mesh_scale,
        max_test_points_before_first=max_test_points_before_first, max_test_points=max_test_points, batch_size=batch_size,
        surface_pointing_tol=surface_pointing_tol, gripper_min_distance=gripper_min_distance,
        np_random=np_random, verbose=verbose
    ):
        grasp_center = grasp_contact_pair.center
        ee_d = grasp_contact_pair.normal1
        # ee_u and ee_v are two vectors that are perpendicular to ee_d
        ee_u = find_orthogonal_vector(ee_d)
        ee_v = np.cross(ee_u, ee_d)

        # enumerate four possible grasp orientations
        for ee_norm1 in [ee_u, ee_v, -ee_u, -ee_v]:
            ee_norm2 = np.cross(ee_d, ee_norm1)
            ee_quat = quaternion_from_axes(ee_norm2, ee_d, ee_norm1)

            qpos = robot.ik(grasp_center, ee_quat)
            if qpos is not None:
                rv = is_collision_free_qpos(robot, qpos, verbose=verbose)
                # robot.set_qpos(qpos)
                # print('is_collision_free_qpos', rv)
                # import ipdb; ipdb.set_trace()

                if rv:
                    yield GraspParameter(
                        contact_pair=grasp_contact_pair,
                        robot_ee_pose=(grasp_center, ee_quat), robot_qpos=qpos
                    )
                elif verbose:
                    print('    gripper pos', grasp_center)
                    print('    gripper quat', ee_quat)
                    print('    skip: collision')


def calc_grasp_approach_ee_pose_trajectory(grasp_parameter: GraspParameter, pregrasp_distance: float = 0.1) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Given the name of the object, the grasp pose, generate a trajectory of 6D poses that the robot should follow to grasp the object."""
    grasp_pos, grasp_quat = grasp_parameter.robot_ee_pose
    trajectory = [(grasp_pos + rotate_vector([0, 0, -pregrasp_distance], grasp_quat), grasp_quat)]
    for i in reversed(range(10)):
        trajectory.append((
            grasp_pos + rotate_vector([0, 0, -pregrasp_distance / 10 * i], grasp_quat),
            grasp_quat
        ))

    return trajectory


class BimanualGraspParameter(NamedTuple):
    contact_pair: GraspContactPairProposal

    pre_contact_distance: float
    post_contact_distance: float

    robot1_ee_pose: Tuple[np.ndarray, np.ndarray]
    robot1_qpos: np.ndarray
    robot2_ee_pose: Tuple[np.ndarray, np.ndarray]
    robot2_qpos: np.ndarray


def gen_bimanual_grasp_parameter(
    planning_world: PlanningWorldInterface, robot1: SingleArmMotionPlanningInterface, robot2: SingleArmMotionPlanningInterface,
    object_id: int, support_id: int, gripper_distance: float, *,
    ee_contact_side: str = 'front',
    pre_contact_distance: float = 0.05, post_contact_distance: float = 0.01,
    gripper_min_distance: float = 0.0001,
    surface_pointing_tol: float = 0.9,
    mesh_filename: Optional[str] = None, mesh_scale: float = 1.0,
    max_test_points_before_first: int = 250, max_test_points: int = 100000000, batch_size: int = 100,
    max_intersection_distance: float = 10,
    max_ik_attempts: int = 5,
    np_random: Optional[np.random.RandomState] = None,
    verbose: bool = False
) -> Iterator[BimanualGraspParameter]:
    def gen_ee_poses(c1, c2, s1 = ee_contact_side, s2 = ee_contact_side):  # contact point 1, contact point 2, side 1, side 2
        for x in pairwise_sample(
            gen_ee_pose_from_contact_point(planning_world, robot1, object_id, c1, distance=pre_contact_distance, z_delta=0.00, side=s1),
            gen_ee_pose_from_contact_point(planning_world, robot2, object_id, c2, distance=pre_contact_distance, z_delta=0.00, side=s2),
            product=True
        ):
            yield c1, c2, *x

    for contact_pair in gen_grasp_contact_pair_with_support_constraint(
        planning_world, object_id, support_id, gripper_distance,
        gripper_min_distance=gripper_min_distance,
        surface_pointing_tol=surface_pointing_tol,
        mesh_filename=mesh_filename, mesh_scale=mesh_scale,
        max_test_points_before_first=max_test_points_before_first, max_test_points=max_test_points, batch_size=batch_size,
        max_intersection_distance=max_intersection_distance,
        np_random=np_random, verbose=verbose
    ):
        for c1, c2, ee_pose1, ee_pose2 in itertools.chain(gen_ee_poses(contact_pair.contact_point1, contact_pair.contact_point2), gen_ee_poses(contact_pair.contact_point2, contact_pair.contact_point1)):
            for qpos1, qpos2 in zip(robot1.gen_ik(*ee_pose1, max_returns=max_ik_attempts), robot2.gen_ik(*ee_pose2, max_returns=max_ik_attempts)):
                with planning_world.checkpoint_world():
                    if robot1.check_collision(qpos1, checkpoint_world_state=False) or robot2.check_collision(qpos2, checkpoint_world_state=False):
                        continue
                yield BimanualGraspParameter(
                    contact_pair=GraspContactPairProposal.from_contact_points(c1, c2),
                    pre_contact_distance=pre_contact_distance,
                    post_contact_distance=post_contact_distance,
                    robot1_ee_pose=ee_pose1, robot1_qpos=qpos1,
                    robot2_ee_pose=ee_pose2, robot2_qpos=qpos2
                )
                break  # If we have found a valid pair of qpos, we can break the loop for zip(robot1.gen_ik, robot2.gen_ik)


def pybullet_visualize_binamual_grasp_parameter(
    planning_world: PlanningWorldInterface, robot1: SingleArmMotionPlanningInterface, robot2: SingleArmMotionPlanningInterface,
    object_id: int, support_id: int, parameter: BimanualGraspParameter
):
    from concepts.dm.crowhat.impl.pybullet.pybullet_planning_world_interface import PyBulletPlanningWorldInterface
    assert isinstance(planning_world, PyBulletPlanningWorldInterface)

    client = planning_world.client

    # Visualize the contact points as a line
    start_pos = parameter.contact_pair.point1 + parameter.contact_pair.normal1 * 0.1
    end_pos = parameter.contact_pair.point2 + parameter.contact_pair.normal2 * 0.1
    client.add_debug_line(start_pos, end_pos, [1, 0, 0], name='grasp_contact_line')

    with client.world.save_world_builtin():
        # Visualize the robot poses
        robot1.set_qpos(parameter.robot1_qpos)
        robot2.set_qpos(parameter.robot2_qpos)

        client.update_viewer_twice()
        client.wait_for_user('Press any key to continue.')


class PlacementParameter(NamedTuple):
    object_id: int
    support_id: int
    target_pos: np.ndarray
    target_quat: np.ndarray
    support_normal: np.ndarray


def gen_placement_parameter(
    planning_world: PlanningWorldInterface,
    target_id: int, support_id: int,
    *,
    retain_target_orientation: bool = False,
    batch_size: int = 100, max_attempts: int = 10000,
    placement_tol: float = 0.03, support_dir_tol: float = 30,
    np_random: Optional[np.random.RandomState] = None,
    verbose: bool = False
) -> Iterator[PlacementParameter]:
    """Sample a placement of the target object on the support object.

    Args:
        planning_world: the planning world.
        robot: the robot interface.
        target_id: the id of the target object.
        support_id: the id of the support object.
        retain_target_orientation: if True, the target object will be placed with the same orientation as the current one.
        batch_size: the number of samples in batch processing.
        max_attempts: the maximum number of attempts for sampling.
        placement_tol: the tolerance for the placement. To check collision, we will place the object at a position that is `placement_tol` away from the support surface and check for collision.
        support_dir_tol: the tolerance for the support direction. We will sample the placement direction such that the support direction is within `support_dir_tol` degree from the gravity direction.
        np_random: the random state.
        verbose: whether to print the sampling information.

    Yields:
        the placement parameters.
    """
    if np_random is None:
        np_random = np.random

    target_mesh = planning_world.get_object_mesh(target_id)
    support_mesh = planning_world.get_object_mesh(support_id)

    nr_batches = int(max_attempts / batch_size)
    for _ in range(nr_batches):
        support_points = _sample_points_uniformly(support_mesh, batch_size, use_triangle_normal=True, seed=np_random.randint(0, 1000000))
        normals = np.asarray(support_points.normals)
        support_indices = np.where(normals.dot(np.array([0, 0, 1])) > np.cos(np.deg2rad(support_dir_tol)))[0]

        target_points = _sample_points_uniformly(target_mesh, batch_size, use_triangle_normal=True, seed=np_random.randint(0, 1000000))

        if retain_target_orientation:
            normals = np.asarray(target_points.normals)
            target_indices = np.where(normals.dot(np.array([0, 0, -1])) > np.cos(np.deg2rad(support_dir_tol)))[0]
        else:
            target_indices = list(range(batch_size))

        all_pair_indices = list(itertools.product(target_indices, support_indices))
        np_random.shuffle(all_pair_indices)
        for target_index, support_index in all_pair_indices:
            target_point_pos = np.asarray(target_points.points[target_index])
            target_point_normal = -np.asarray(target_points.normals[target_index])

            support_point_pos = np.asarray(support_points.points[support_index])
            support_point_normal = np.asarray(support_points.normals[support_index])

            current_target_pos, current_target_quat = planning_world.get_object_pose(target_id)

            # Solve for a quaternion that aligns the tool normal with the target normal
            for rotation_quat in enumerate_quaternion_from_vectors(target_point_normal, support_point_normal, 4):
                # This is the world coordinate for the tool point after rotation.
                new_target_point_pos = current_target_pos + rotate_vector(target_point_pos - current_target_pos, rotation_quat)
                # Now compute the displacement for the tool object
                final_target_pos = support_point_pos - new_target_point_pos + current_target_pos
                final_target_pos += support_point_normal * placement_tol
                final_target_quat = quat_mul(rotation_quat, current_target_quat)

                success = True
                with planning_world.checkpoint_world():
                    planning_world.set_object_pose(target_id, (final_target_pos, final_target_quat))
                    contacts = planning_world.get_contact_points(target_id)

                    for contact in contacts:
                        if contact.body_b != target_id:
                            success = False
                            break

                if success:
                    yield PlacementParameter(
                        object_id=target_id,
                        support_id=support_id,
                        target_pos=final_target_pos,
                        target_quat=final_target_quat,
                        support_normal=support_point_normal
                    )
