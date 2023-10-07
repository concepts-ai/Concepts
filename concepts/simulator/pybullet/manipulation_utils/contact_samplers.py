#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : contact_samplers.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/05/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import itertools
import random
from dataclasses import dataclass
from typing import Optional, Iterable, Sequence, Tuple, List

import numpy as np
import numpy.random as npr
import open3d as o3d

import jacinle
from jacinle.logging import get_logger
from concepts.simulator.pybullet.rotation_utils import get_quaternion_from_axes, find_orthogonal_vector, quat_mul, rotate_vector, enumerate_quaternion_from_vectors
from concepts.simulator.pybullet.components.robot_base import Robot

logger = get_logger(__file__)

__all__ = [
    'GraspParameter', 'sample_grasp', 'gen_grasp_trajectory',
    'PushParameter', 'sample_push_with_support', 'gen_push_trajectory',
    'IndirectPushParameter', 'sample_indirect_push_with_support',
    'PlacementParameter',
    'qpos_trajectory_from_poses', 'qpos_trajectory_from_poses_pybullet',
    'collision_free_qpos', 'collision_free_pos', 'collision_free_qpos_trajectory',
    'mesh_line_intersect', 'get_single_contact_normal'
]


@dataclass
class GraspParameter(object):
    point1: np.ndarray
    normal1: np.ndarray
    point2: np.ndarray
    normal2: np.ndarray
    ee_pos: np.ndarray
    ee_quat: np.ndarray
    qpos: np.ndarray


def sample_grasp(
    robot: Robot, object_id: int,
    gripper_distance: float, max_intersection_distance: float = 10,
    mesh_filename: Optional[str] = None, mesh_scale: float = 1.0,
    verbose: bool = False, max_test_points_before_first: int = 250, max_test_points: int = 100000000, batch_size: int = 100,
    surface_pointing_tol: float = 0.9, min_point_distance: float = 0.0001
) -> Iterable[GraspParameter]:
    """Given the name of the object, sample a 6D grasp pose. Before calling this function, we should make sure that the gripper is open."""

    mesh = robot.world.get_mesh(object_id, zero_center=False, mesh_filename=mesh_filename, mesh_scale=mesh_scale)
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    # print the x, y, z center of the mesh
    # print('mesh center', mesh.get_center())
    # bounding box
    # print('mesh bounding box', mesh.get_axis_aligned_bounding_box())

    found = False
    nr_test_points_before_first = 0

    for _ in range(int(max_test_points / batch_size)):
        # TODO(Jiayuan Mao @ 2023/01/02): accelerate the computation.
        pcd = mesh.sample_points_uniformly(batch_size, use_triangle_normal=True)
        # pcd = mesh.sample_points_poisson_disk(batch_size, use_triangle_normal=True)

        indices = list(range(len(pcd.points)))
        random.shuffle(indices)
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
            if np.linalg.norm(other_point - point) < min_point_distance:
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
            grasp_normal = normal

            if grasp_distance > gripper_distance:
                if verbose:
                    print('  skip: too far')
                continue

            ee_d = grasp_normal
            # ee_u and ee_v are two vectors that are perpendicular to ee_d
            ee_u = find_orthogonal_vector(ee_d)
            ee_v = np.cross(ee_u, ee_d)

            # if verbose:
            #     print('  grasp_center', grasp_center, 'grasp_distance', grasp_distance)
            #     print('  grasp axes:\n', np.array([ee_d, ee_u, ee_v]))

            # enumerate four possible grasp orientations
            for ee_norm1 in [ee_u, ee_v, -ee_u, -ee_v]:
                ee_norm2 = np.cross(ee_d, ee_norm1)
                ee_quat = get_quaternion_from_axes(ee_norm2, ee_d, ee_norm1)

                qpos = robot.ikfast(grasp_center, ee_quat, max_attempts=100, error_on_fail=False)
                # qpos = ctx.robot._solve_ik(grasp_center, ee_quat, force=True)
                if qpos is not None:
                    rv = collision_free_qpos(robot, qpos, verbose=verbose)
                    # with robot.world.save_body(robot.get_robot_body_id()):
                    #     robot.set_qpos(qpos)
                    #     robot.client.wait_for_duration(0.1)
                    #     # visualize the mesh and point and other_point as two red spheres
                    #     o3d.visualization.draw_geometries([
                    #         mesh,
                    #         o3d.geometry.TriangleMesh.create_sphere(0.1).translate(point),
                    #         o3d.geometry.TriangleMesh.create_sphere(0.1).translate(other_point),
                    #     ])

                    #     input(f'testing grasp, cfree={rv}')

                    if rv:
                        found = True
                        yield GraspParameter(
                            point1=point, normal1=normal,
                            point2=other_point, normal2=other_normal,
                            ee_pos=grasp_center, ee_quat=ee_quat,
                            qpos=qpos
                        )
                    elif verbose:
                        print('    gripper pos', grasp_center)
                        print('    gripper quat', ee_quat)
                        print('    skip: collision')

        if not found and nr_test_points_before_first > max_test_points_before_first:
            if verbose:
                logger.warning(f'Failed to find a grasp after {nr_test_points_before_first} points tested.')
            return


def gen_grasp_trajectory(grasp_pos: np.ndarray, grasp_quat: np.ndarray, height: float = 0.1) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Given the name of the object, the grasp pose, generate a trajectory of 6D poses that the robot should follow to grasp the object."""

    trajectory = [(grasp_pos + rotate_vector([0, 0, -height], grasp_quat), grasp_quat)]
    for i in reversed(range(10)):
        trajectory.append((
            grasp_pos + rotate_vector([0, 0, -height / 10 * i], grasp_quat),
            grasp_quat
        ))

    return trajectory


@dataclass
class PushParameter(object):
    push_pos: np.ndarray
    push_dir: np.ndarray
    distance: float
    ee_quat: np.ndarray


def sample_push_with_support(robot: Robot, object_id: int, support_object_id: int, max_attempts: int = 1000, batch_size: int = 100, verbose=False) -> Iterable[PushParameter]:
    mesh = robot.world.get_mesh(object_id, zero_center=False)

    nr_batches = int(max_attempts / batch_size)
    feasible_point_indices = list()
    for _ in range(nr_batches):
        pcd = mesh.sample_points_uniformly(batch_size, use_triangle_normal=True)

        # get the contact points between the object and the support object
        contact_normal = get_single_contact_normal(robot, object_id, support_object_id)

        # filter out the points that are not on the contact plane
        feasible_point_cond = np.abs(np.asarray(pcd.normals).dot(contact_normal)) < 0.02
        feasible_point_indices = np.where(feasible_point_cond)[0]

        # print(f'Found {len(feasible_point_indices)} feasible points.')

        if len(feasible_point_indices) == 0:
            continue

        # o3d.visualization.draw([pcd.select_by_index(feasible_point_indices), mesh])

        npr.shuffle(feasible_point_indices)
        rows = list()
        for index in feasible_point_indices:
            rows.append((index, pcd.points[index], -pcd.normals[index]))

        if verbose:
            import tabulate
            jacinle.log_function.print(tabulate.tabulate(rows, headers=['index', 'point', 'normal']))

        # create a new point cloud
        for index in feasible_point_indices:
            if verbose:
                jacinle.log_function.print('sample_push_with_support', 'point', pcd.points[index], 'normal', -pcd.normals[index])
            yield PushParameter(np.asarray(pcd.points[index]), -np.asarray(pcd.normals[index]), 0.1, robot.get_ee_home_quat())

    if len(feasible_point_indices) == 0:
        raise ValueError(f'No feasible points for {object_id} on {support_object_id} after {nr_batches * batch_size} attempts.')


def gen_push_trajectory(push_pos: np.ndarray, push_dir: np.ndarray, push_distance: float, quat: np.ndarray, prepush_distance: float = 0.1) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Given the name of the object, the push pose, the push direction, and the push distance, generate a trajectory of 6D poses that the robot should follow to push the object."""

    push_pos += (0, 0, 0.010)
    real_start_pos = push_pos - prepush_distance * push_dir / np.linalg.norm(push_dir)
    real_end_pos = push_pos + push_dir * push_distance

    trajectory = list()
    for i in range(10 + 1):
        pos = real_start_pos + (real_end_pos - real_start_pos) * i / 10
        trajectory.append((pos, quat))

    return trajectory


@dataclass
class IndirectPushParameter(object):
    target_push_pos: np.ndarray
    target_push_dir: np.ndarray

    tool_pos: np.ndarray
    tool_quat: np.ndarray

    tool_point_pos: np.ndarray
    tool_point_normal: np.ndarray

    ee_pos: np.ndarray = None
    ee_quat: np.ndarray = None

    prepush_distance: float = 0.05
    push_distance: float = 0.1

    @property
    def total_push_distance(self):
        return self.prepush_distance + self.push_distance


def sample_indirect_push_with_support(
    robot: Robot, target_id: int, tool_id: int, support_object_id: int,
    prepush_distance: float = 0.05,
    max_attempts: int = 10000, batch_size: int = 100, filter_push_dir: Optional[np.ndarray] = None,
    push_distance_distribution: Sequence[float] = (0.1, 0.15), push_distance_sample: bool = False,
    contact_normal_tol: float = 0.01, verbose: bool = False
) -> Iterable[IndirectPushParameter]:
    """Sample a push of the target object using the tool object. The target object should be placed on the support object.

    Args:
        robot: the robot.
        target_id: the PyBullet body id of the target object.
        tool_id: the PyBullet body id of the tool object.
        support_object_id: the PyBullet body id of the support object.
        prepush_distance: the distance to pre-push the tool object before the actual push.
        max_attempts: the maximum number of attempts for sampling.
        batch_size: the number of samples in batch processing.
        filter_push_dir: if specified, the push direction will be filtered to be within 0.2 cosine distance from this direction.
        push_distance_distribution: the distribution of the push distance. If `push_distance_sample` is True, this will be used to sample the push distance.
        push_distance_sample: whether to sample the push distance from the distribution. If False, the function will enumerate the push distances in the distribution.
        contact_normal_tol: the tolerance for the contact normal. We will sample the push direction such that the contact normal is within `contact_normal_tol` degree from the gravity direction.
        verbose: whether to print the sampling information.
    """
    target_mesh = robot.world.get_mesh(target_id, zero_center=False)
    tool_mesh = robot.world.get_mesh(tool_id, zero_center=False)

    nr_batches = int(max_attempts / batch_size)
    contact_normal = get_single_contact_normal(robot, target_id, support_object_id)

    for _ in range(nr_batches):
        target_pcd = target_mesh.sample_points_uniformly(batch_size, use_triangle_normal=True)
        tool_pcd = tool_mesh.sample_points_uniformly(batch_size, use_triangle_normal=True)

        # feasible_target_point_cond = np.abs(np.asarray(target_pcd.normals).dot(contact_normal)) < 0.01
        # 0.1 for real demo.
        feasible_target_point_cond = np.abs(np.asarray(target_pcd.normals).dot(contact_normal)) < contact_normal_tol

        if filter_push_dir is not None:
            feasible_target_point_cond = np.logical_and(
                feasible_target_point_cond,
                np.asarray(target_pcd.normals, dtype=np.float32).dot(-filter_push_dir) > 0.8
            )

        feasible_target_point_indices = np.where(feasible_target_point_cond)[0]

        all_index_pairs = list(itertools.product(feasible_target_point_indices, range(batch_size)))
        random.shuffle(all_index_pairs)
        for target_index, tool_index in all_index_pairs:
            target_point_pos = np.asarray(target_pcd.points[target_index])
            target_point_normal = -np.asarray(target_pcd.normals[target_index])  # point inside

            tool_point_pos = np.asarray(tool_pcd.points[tool_index])
            tool_point_normal = np.asarray(tool_pcd.normals[tool_index])  # point outside (towards the tool)

            current_tool_pos, current_tool_quat = robot.world.get_body_state_by_id(tool_id).get_transformation()

            # Solve for a quaternion that aligns the tool normal with the target normal
            for rotation_quat in enumerate_quaternion_from_vectors(tool_point_normal, target_point_normal, 4):
                # This is the world coordinate for the tool point after rotation.
                new_tool_point_pos = current_tool_pos + rotate_vector(tool_point_pos - current_tool_pos, rotation_quat)
                # Now compute the displacement for the tool object
                final_tool_pos = target_point_pos - new_tool_point_pos + current_tool_pos
                final_tool_pos -= target_point_normal * prepush_distance
                final_tool_quat = quat_mul(rotation_quat, current_tool_quat)

                success = True
                with robot.world.save_body(tool_id):
                    robot.world.set_body_state2_by_id(tool_id, final_tool_pos, final_tool_quat)
                    contacts = robot.world.get_contact(tool_id, update=True)

                    for contact in contacts:
                        if contact.body_b != tool_id:
                            success = False
                            break

                if success:
                    if push_distance_sample:
                        distances = [npr.choice(push_distance_distribution)]
                    else:
                        distances = push_distance_distribution
                    kwargs = dict(
                        target_push_pos=target_point_pos,
                        target_push_dir=target_point_normal,
                        tool_pos=final_tool_pos,
                        tool_quat=final_tool_quat,
                        tool_point_pos=rotate_vector(tool_point_pos - current_tool_pos, rotation_quat) + final_tool_pos,
                        tool_point_normal=rotate_vector(tool_point_normal, rotation_quat),
                        prepush_distance=prepush_distance
                    )
                    for distance in distances:
                        yield IndirectPushParameter(**kwargs, push_distance=distance)


@dataclass
class PlacementParameter(object):
    target_pos: np.ndarray
    target_quat: np.ndarray

    support_normal: np.ndarray


def sample_placement(
    robot: Robot, target_id: int, support_id: int,
    retain_target_orientation: bool = False,
    batch_size: int = 100, max_attempts: int = 10000,
    placement_tol: float = 0.03, support_dir_tol: float = 30,
    verbose: bool = False
) -> Iterable[PlacementParameter]:
    """Sample a placement of the target object on the support object.

    Args:
        robot: the robot.
        target_id: the id of the target object.
        support_id: the id of the support object.
        retain_target_orientation: if True, the target object will be placed with the same orientation as the current one.
        batch_size: the number of samples in batch processing.
        max_attempts: the maximum number of attempts for sampling.
        placement_tol: the tolerance for the placement. To check collision, we will place the object at a position that is `placement_tol` away from the support surface and check for collision.
        support_dir_tol: the tolerance for the support direction. We will sample the placement direction such that the support direction is within `support_dir_tol` degree from the gravity direction.

    Yields:
        the placement parameters.
    """
    target_mesh = robot.world.get_mesh(target_id, zero_center=False)
    support_mesh = robot.world.get_mesh(support_id, zero_center=False)

    nr_batches = int(max_attempts / batch_size)
    for _ in range(nr_batches):
        support_points = support_mesh.sample_points_uniformly(batch_size, use_triangle_normal=True)
        normals = np.asarray(support_points.normals)
        support_indices = np.where(normals.dot(np.array([0, 0, 1])) > np.cos(np.deg2rad(support_dir_tol)))[0]

        # support_points_inner = o3d.geometry.PointCloud()
        # support_points_inner.points = o3d.utility.Vector3dVector(np.asarray(support_points.points)[support_indices])
        # support_points_inner.normals = o3d.utility.Vector3dVector(np.asarray(support_points.normals)[support_indices])
        # o3d.visualization.draw_geometries([support_mesh, support_points_inner])

        target_points = target_mesh.sample_points_uniformly(batch_size, use_triangle_normal=True)

        if retain_target_orientation:
            normals = np.asarray(target_points.normals)
            target_indices = np.where(normals.dot(np.array([0, 0, -1])) > np.cos(np.deg2rad(support_dir_tol)))[0]
        else:
            target_indices = list(range(batch_size))

        # o3d.visualization.draw_geometries([target_mesh, target_points])

        all_pair_indices = list(itertools.product(target_indices, support_indices))
        random.shuffle(all_pair_indices)
        for target_index, support_index in all_pair_indices:
            target_point_pos = np.asarray(target_points.points[target_index])
            target_point_normal = -np.asarray(target_points.normals[target_index])

            support_point_pos = np.asarray(support_points.points[support_index])
            support_point_normal = np.asarray(support_points.normals[support_index])

            current_target_pos, current_target_quat = robot.world.get_body_state_by_id(target_id).get_transformation()

            # Solve for a quaternion that aligns the tool normal with the target normal
            for rotation_quat in enumerate_quaternion_from_vectors(target_point_normal, support_point_normal, 4):
                # This is the world coordinate for the tool point after rotation.
                new_target_point_pos = current_target_pos + rotate_vector(target_point_pos - current_target_pos, rotation_quat)
                # Now compute the displacement for the tool object
                final_target_pos = support_point_pos - new_target_point_pos + current_target_pos
                final_target_pos += support_point_normal * placement_tol
                final_target_quat = quat_mul(rotation_quat, current_target_quat)

                # print('final_target_pos', final_target_pos, 'final_target_quat', final_target_quat)

                success = True
                with robot.world.save_body(target_id):
                    robot.world.set_body_state2_by_id(target_id, final_target_pos, final_target_quat)
                    contacts = robot.world.get_contact(target_id, update=True)

                    for contact in contacts:
                        if contact.body_b != target_id:
                            # print('collision ...', 'between', robot.world.get_body_name(contact.body_a), 'and', robot.world.get_body_name(contact.body_b))
                            # input('press enter to continue')
                            success = False
                            break

                if success:
                    yield PlacementParameter(
                        target_pos=final_target_pos,
                        target_quat=final_target_quat,
                        support_normal=support_point_normal
                    )


def qpos_trajectory_from_poses(robot: Robot, trajectory: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[bool, List[np.ndarray]]:
    """Given a list of 6D poses, generate a list of qpos that the robot should follow to follow the trajectory."""

    qpos_trajectory = list()
    last_qpos = robot.ikfast(trajectory[0][0], trajectory[0][1], error_on_fail=False)
    if last_qpos is None:
        return False, qpos_trajectory
    qpos_trajectory.append(last_qpos)

    success = True
    for this_pos, this_quat in trajectory:
        qpos = robot.ikfast(this_pos, this_quat, max_distance=0.1, last_qpos=last_qpos, error_on_fail=False)
        if qpos is None:
            return False, qpos_trajectory
        qpos_trajectory.append(qpos)
        last_qpos = qpos

    return success, qpos_trajectory


def qpos_trajectory_from_poses_pybullet(robot: Robot, trajectory: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[bool, List[np.ndarray]]:
    qpos_trajectory = list()
    for pos, quat in trajectory:
        qpos = robot.ik_pybullet(pos, quat)
        if qpos is None:
            return False, qpos_trajectory
        qpos_trajectory.append(qpos)
    return True, qpos_trajectory


def collision_free_qpos(robot: Robot, qpos: np.ndarray, exclude: Optional[List[int]] = None, verbose: bool = False) -> bool:
    """Check whether the given qpos is collision free. The function also accepts a list of object ids to exclude (e.g., the object in hand).

    Args:
        robot: the robot.
        qpos: the qpos to check.
        exclude: the object ids to exclude.
        verbose: whether to print the collision information.

    Returns:
        True if the qpos is collision free.
    """
    with robot.world.save_body(robot.get_robot_body_id()):
        robot.set_qpos(qpos)
        contacts = robot.world.get_contact(robot.get_robot_body_id(), update=True)
        if exclude is not None:
            for c in contacts:
                if c.body_b not in exclude and c.body_b != c.body_a:
                    if verbose:
                        print(f'  collision_free_qpos: collide bewteen {robot.world.body_names[c.body_a]} and {robot.world.body_names[c.body_b]}')
                    return False
        else:
            for c in contacts:
                if c.body_b != c.body_a:
                    if verbose:
                        print(f'  collision_free_qpos: collide bewteen {robot.world.body_names[c.body_a]} and {robot.world.body_names[c.body_b]}')
                    return False
        return True


def collision_free_pos(robot: Robot, pos: np.ndarray, quat: Optional[np.ndarray] = None, exclude: Optional[List[int]] = None, verbose: bool = False) -> bool:
    """Check whether there is a qpos at the givne pose that is collision free. The function also accepts a list of object ids to exclude (e.g., the object in hand).
    This function is not recommended to use, since it is not fully reproducible (there will be multiple qpos that can achieve the same pose).

    Args:
        robot: the robot.
        pos: the position to check.
        quat: the quaternion to check. If not specified, the home quaternion will be used.
        exclude: the object ids to exclude.
        verbose: whether to print the collision information.

    Returns:
        True if there is a collision free qpos.
    """

    if quat is None:
        quat = robot.get_ee_home_quat()
    qpos = robot.ikfast(pos, quat)
    if qpos is None:
        return False
    return collision_free_qpos(robot, qpos, exclude=exclude, verbose=verbose)


def collision_free_qpos_trajectory(robot: Robot, qpos_trajectory: List[np.ndarray], exclude: Optional[List[int]] = None, verbose: bool = False) -> bool:
    for qpos in qpos_trajectory:
        if not collision_free_qpos(robot, qpos, exclude=exclude, verbose=verbose):
            return False
    return True


def mesh_line_intersect(t_mesh: o3d.t.geometry.TriangleMesh, ray_origin: np.ndarray, ray_direction: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Intersects a ray with a mesh.

    Args:
        t_mesh: the mesh to intersect with.
        ray_origin: the origin of the ray.
        ray_direction: the direction of the ray.

    Returns:
        A tuple of (point, normal) if an intersection is found, None otherwise.
    """

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)

    ray = o3d.core.Tensor.from_numpy(np.array(
        [[ray_origin[0], ray_origin[1], ray_origin[2], ray_direction[0], ray_direction[1], ray_direction[2]]],
        dtype=np.float32
    ))
    result = scene.cast_rays(ray)

    # no intersection.
    if result['geometry_ids'][0] == scene.INVALID_ID:
        return None

    inter_point = np.asarray(ray_origin) + np.asarray(ray_direction) * result['t_hit'][0].item()
    inter_normal = result['primitive_normals'][0].numpy()
    return inter_point, inter_normal


def get_single_contact_normal(robot: Robot, object_id: int, support_object_id: int, deviation_tol: float = 0.05) -> np.ndarray:
    body_names = robot.world.body_names.int_to_string
    object_name, support_name = body_names[object_id], body_names[support_object_id]
    contacts = robot.world.get_contact(object_id, support_object_id)

    if len(contacts) == 0:
        # TODO(Jiayuan Mao @ 2023/03/15): find a better way to configure this.
        robot.client.step(1)
        contacts = robot.world.get_contact(object_id, support_object_id)

    if len(contacts) == 0:
        raise ValueError(f'No contact between {object_name} and {support_name}.')

    contact_normals = np.array([c.contact_normal_on_b for c in contacts])
    contact_normal_avg = np.mean(contact_normals, axis=0)
    contact_normal_avg /= np.linalg.norm(contact_normal_avg)

    deviations = np.abs(1 - contact_normals.dot(contact_normal_avg) / np.linalg.norm(contact_normals, axis=1))
    if np.max(deviations) > deviation_tol:
        raise ValueError(
            f'Contact normals of {object_name} and {support_name} are not consistent. This is likely due to multiple contact points.\n'
            f'  Contact normals: {contact_normals}\n  Deviations: {deviations}.'
        )

    return contact_normal_avg

