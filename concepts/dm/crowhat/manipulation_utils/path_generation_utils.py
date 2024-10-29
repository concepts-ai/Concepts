#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : path_generation_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/07/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
from typing import Optional, Union, Iterator, Sequence, Tuple, List

from concepts.math.rotationlib_xyzw import normalize_vector, slerp
from concepts.math.frame_utils_xyzw import get_transform_a_to_b
from concepts.utils.typing_utils import Vec3f, Vec4f, VecNf
from concepts.dm.crowhat.world.manipulator_interface import SingleArmMotionPlanningInterface
from concepts.dm.crowhat.manipulation_utils.pose_utils import angle_distance

__all__ = [
    # Grasping related functions.
    'calc_grasp_pose_from_current_object_pose_and_ee_pose',
    # Path planning from qpos and ee path.
    'gen_collision_free_qpos_path_from_current_qpos_and_target_qpos',
    'calc_interpolated_qpos_path_from_current_qpos_and_target_qpos',
    'is_collision_free_qpos', 'is_collision_free_ee_pose', 'is_collision_free_qpos_path',
    # Collision-free path planning from end-effector path and current qpos.
    'gen_qpos_path_from_ee_path',
    'gen_collision_free_qpos_path_from_current_qpos_and_ee_path',
    'gen_collision_free_qpos_path_from_current_qpos_and_ee_linear_path',
    'gen_collision_free_qpos_path_from_current_qpos_and_ee_pose',
    'gen_qpos_path_from_current_qpos_and_ee_pose',
    # Smooth path generation.
    'calc_smooth_qpos_path_from_qpos_path', 'wrap_iterator_smooth_qpos_path',
    'calc_smooth_ee_path_from_ee_path',
]


def calc_grasp_pose_from_current_object_pose_and_ee_pose(robot: SingleArmMotionPlanningInterface, object_id: int, ee_pos: Vec3f, ee_dir: Vec3f = (0., 0., -1.), ee_dir2: Vec3f = (1., 0., 0.)) -> Tuple[Vec3f, Vec4f]:
    """Compute the grasp pose from the current object pose and the desired ee pose.

    Args:
        robot: the robot.
        object_id: the object id.
        ee_pos: the desired ee position.
        ee_dir: the desired ee direction.
        ee_dir2: the desired ee direction 2.
    """

    ee_quat = robot.calc_ee_quat_from_vectors(ee_dir, ee_dir2)
    obj_pos, obj_quat = robot.planning_world.get_object_pose(object_id)
    ee_to_object = get_transform_a_to_b(ee_pos, ee_quat, obj_pos, obj_quat)
    return ee_to_object


def gen_collision_free_qpos_path_from_current_qpos_and_target_qpos(
    robot: SingleArmMotionPlanningInterface, target_qpos: np.ndarray, ignored_collision_bodies: Optional[List[int]] = None,
    return_smooth_path: bool = False,
    verbose: bool = False
) -> Optional[List[np.ndarray]]:
    """Generate a collision-free path from the current qpos to reach the target qpos.

    Args:
        robot: the robot.
        target_qpos: the target qpos.
        ignored_collision_bodies: the object ids to ignore in collision checking.
        return_smooth_path: whether to return the smooth path.
        verbose: whether to print additional information.

    Returns:
        the qpos path if successful.
    """
    rv, path = robot.rrt_collision_free(robot.get_qpos(), target_qpos, smooth_fine_path=True, ignored_collision_bodies=ignored_collision_bodies, verbose=verbose)
    if not rv:
        return None
    return calc_smooth_qpos_path_from_qpos_path(robot, path) if return_smooth_path else path


def calc_interpolated_qpos_path_from_current_qpos_and_target_qpos(robot: SingleArmMotionPlanningInterface, target_qpos: np.ndarray) -> List[np.ndarray]:
    """Generate a linearly interpolated path from the current qpos to reach the target qpos.

    Args:
        robot: the robot.
        target_qpos: the target qpos.

    Returns:
        the qpos path.
    """
    return robot.get_configuration_space().gen_path(robot.get_qpos(), target_qpos)[1]


def is_collision_free_qpos(robot: SingleArmMotionPlanningInterface, qpos: np.ndarray, exclude: Optional[List[int]] = None, verbose: bool = False) -> bool:
    """Check whether the given qpos is collision free. The function also accepts a list of object ids to exclude (e.g., the object in hand).

    Args:
        robot: the robot.
        qpos: the qpos to check.
        exclude: the object ids to exclude.
        verbose: whether to print the collision information.

    Returns:
        True if the qpos is collision free.
    """
    return not robot.check_collision(qpos, ignore_self_collision=True, ignored_collision_bodies=exclude)


def is_collision_free_ee_pose(robot: SingleArmMotionPlanningInterface, pos: np.ndarray, quat: Optional[np.ndarray] = None, exclude: Optional[List[int]] = None, verbose: bool = False) -> bool:
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
        quat = robot.get_ee_default_quat()
    qpos = robot.ik(pos, quat)
    if qpos is None:
        return False
    return is_collision_free_qpos(robot, qpos, exclude=exclude, verbose=verbose)


def is_collision_free_qpos_path(robot: SingleArmMotionPlanningInterface, qpos_trajectory: List[np.ndarray], exclude: Optional[List[int]] = None, verbose: bool = False) -> bool:
    for qpos in qpos_trajectory:
        if not is_collision_free_qpos(robot, qpos, exclude=exclude, verbose=verbose):
            return False
    return True


def gen_qpos_path_from_ee_path(robot: SingleArmMotionPlanningInterface, trajectory: List[Tuple[np.ndarray, np.ndarray]], max_pairwise_distance: float = 0.1, verbose: bool = False) -> Tuple[bool, List[np.ndarray]]:
    """Given a list of 6D poses, generate a list of qpos that the robot should follow to follow the trajectory.

    Args:
        robot: the robot.
        trajectory: the trajectory of 6D poses.
        max_pairwise_distance: the maximum distance between two consecutive qpos. This is the L2 distance in the configuration space.
        verbose: whether to print additional information.

    Returns:
        a tuple of (success, qpos_trajectory).
        If success is False, the qpos_trajectory will contain the qpos that have been successfully computed.
    """

    qpos_trajectory = list()
    last_qpos = robot.ik(trajectory[0][0], trajectory[0][1])
    if last_qpos is None:
        if verbose:
            print('qpos_trajectory_from_poses: failed to compute the first qpos.')
        return False, qpos_trajectory
    qpos_trajectory.append(last_qpos)

    success = True
    for i, (this_pos, this_quat) in enumerate(trajectory):
        qpos = robot.ik(this_pos, this_quat, qpos=last_qpos, max_distance=max_pairwise_distance)
        if qpos is None:
            if verbose:
                print(f'qpos_trajectory_from_poses: failed to compute the qpos at step #{i+1}.')
            return False, qpos_trajectory
        qpos_trajectory.append(qpos)
        last_qpos = qpos

    return success, qpos_trajectory


def gen_collision_free_qpos_path_from_current_qpos_and_ee_path(
    robot: SingleArmMotionPlanningInterface, waypoints: Sequence[Tuple[Vec3f, Vec4f]], *,
    nr_ik_attempts: int = 1,
    collision_free_planning: bool = True, collision_free_planning_first: bool = False, ignored_collision_bodies: Optional[List[int]] = None,
    return_smooth_path: bool = False, generator: bool = False, verbose: bool = False,
) -> Union[Iterator[List[VecNf]], Optional[List[VecNf]]]:
    def dfs(remaining_waypoints, current_qpos: VecNf, is_first: bool = False) -> Iterator[List[VecNf]]:
        if len(remaining_waypoints) == 0:
            yield list()
            return
        target_pos, target_quat = remaining_waypoints[0]
        for _ in range(nr_ik_attempts):
            solution = robot.ik(target_pos, target_quat, qpos=current_qpos)
            if verbose:
                print(f'Generating IK (remaining waypoints: {len(remaining_waypoints)}): {solution is not None}')
            if solution is None:
                continue
            if robot.check_collision(solution, ignored_collision_bodies=ignored_collision_bodies):
                if verbose:
                    print(f'Collision detected (remaining waypoints: {len(remaining_waypoints)}): {solution is not None}')
                continue

            if collision_free_planning and (not collision_free_planning_first or is_first):
                rv, path = robot.rrt_collision_free(current_qpos, solution, smooth_fine_path=True, ignored_collision_bodies=ignored_collision_bodies, verbose=verbose)
                if verbose:
                    print(f'Generating path (remaining waypoints: {len(remaining_waypoints)}): {rv} {len(path) if path is not None else 0}')
                if rv:
                    for remaining_path in dfs(remaining_waypoints[1:], path[-1]):
                        yield path + remaining_path
            else:
                for remaining_path in dfs(remaining_waypoints[1:], solution):
                    yield [solution] + remaining_path

    if generator:
        if return_smooth_path:
            return wrap_iterator_smooth_qpos_path(robot, dfs(waypoints, robot.get_qpos(), is_first=True))
        return dfs(waypoints, robot.get_qpos(), is_first=True)

    for x in dfs(waypoints, robot.get_qpos(), is_first=True):
        return calc_smooth_qpos_path_from_qpos_path(robot, x) if return_smooth_path else x
    return None


def gen_collision_free_qpos_path_from_current_qpos_and_ee_linear_path(
    robot: SingleArmMotionPlanningInterface, target_pos1, target_pos2, ee_dir: Optional[Vec3f] = (0, 0, -1), ee_dir2: Optional[Vec3f] = (1, 0, 0), nr_waypoints: int = 10, *,
    nr_ik_attempts: int = 1,
    collision_free_planning: bool = True, collision_free_planning_first: bool = False, ignored_collision_bodies: Optional[List[int]] = None,
    return_smooth_path: bool = False, generator: bool = False, verbose: bool = False
):
    """Generate a collision-free Cartesian path from the current qpos to follow a straight line from target_pos1 to target_pos2."""
    quat = robot.calc_ee_quat_from_vectors(ee_dir, ee_dir2)
    steps = np.linspace(target_pos1, target_pos2, nr_waypoints)
    waypoints = [(step, quat) for step in steps]

    return gen_collision_free_qpos_path_from_current_qpos_and_ee_path(
        robot, waypoints, ignored_collision_bodies=ignored_collision_bodies,
        nr_ik_attempts=nr_ik_attempts, collision_free_planning=collision_free_planning, collision_free_planning_first=collision_free_planning_first,
        return_smooth_path=return_smooth_path, generator=generator, verbose=verbose
    )


def gen_collision_free_qpos_path_from_current_qpos_and_ee_pose(
    robot: SingleArmMotionPlanningInterface, target_pos: Vec3f, target_quat: Optional[Vec4f] = None, ee_dir: Optional[Vec3f] = (0, 0, -1), ee_dir2: Optional[Vec3f] = (1, 0, 0), *,
    nr_ik_attempts: int = 1,
    collision_free_planning: bool = True, collision_free_planning_first: bool = False, ignored_collision_bodies: Optional[List[int]] = None,
    return_smooth_path: bool = False, generator: bool = False, verbose: bool = False
):
    """Generate a collision-free path from the current qpos to reach the target_pos."""
    if target_quat is None:
        target_quat = robot.calc_ee_quat_from_vectors(ee_dir, ee_dir2)
    waypoints = [(target_pos, target_quat)]

    return gen_collision_free_qpos_path_from_current_qpos_and_ee_path(
        robot, waypoints, ignored_collision_bodies=ignored_collision_bodies, nr_ik_attempts=nr_ik_attempts,
        collision_free_planning=collision_free_planning, collision_free_planning_first=collision_free_planning_first,
        return_smooth_path=return_smooth_path, generator=generator, verbose=verbose
    )


def gen_qpos_path_from_current_qpos_and_ee_pose(
    robot: SingleArmMotionPlanningInterface, target_pos: Vec3f, target_quat: Optional[Vec4f] = None, ee_dir: Optional[Vec3f] = (0, 0, -1), ee_dir2: Optional[Vec3f] = (1, 0, 0), *,
    nr_ik_attempts: int = 1,
    return_smooth_path: bool = False, generator: bool = False, verbose: bool = False
):
    """Generate a collision-free path from the current qpos to reach the target_pos and target_quat."""
    if target_quat is None:
        target_quat = robot.calc_ee_quat_from_vectors(ee_dir, ee_dir2)
    waypoints = [(target_pos, target_quat)]

    return gen_collision_free_qpos_path_from_current_qpos_and_ee_path(
        robot, waypoints, nr_ik_attempts=nr_ik_attempts, collision_free_planning=False,
        return_smooth_path=return_smooth_path, generator=generator, verbose=verbose
    )


def calc_smooth_qpos_path_from_qpos_path(robot: SingleArmMotionPlanningInterface, qt: Optional[List[VecNf]], skip_first: bool = True) -> Optional[List[VecNf]]:
    if qt is None:
        return None

    if not skip_first:
        qt = qt.copy()
        qt.insert(0, robot.get_qpos())

    smooth_qt = [qt[0]]
    cspace = robot.get_configuration_space()
    for qpos1, qpos2 in zip(qt[:-1], qt[1:]):
        smooth_qt.extend(cspace.gen_path(qpos1, qpos2)[1][1:])
    return smooth_qt


def wrap_iterator_smooth_qpos_path(robot: SingleArmMotionPlanningInterface, qt_iterator: Iterator[List[VecNf]]) -> Iterator[List[VecNf]]:
    for qt in qt_iterator:
        yield calc_smooth_qpos_path_from_qpos_path(robot, qt)


def calc_smooth_ee_path_from_ee_path(ee_path: Sequence[Tuple[Vec3f, Vec4f]], max_linear_velocity: float = 0.1, max_angular_velocity: float = np.pi / 4, fps: int = 60) -> List[Tuple[Vec3f, Vec4f]]:
    max_linear_velocity = max_linear_velocity / fps
    max_angular_velocity = max_angular_velocity / fps

    smooth_ee_path = [ee_path[0]]
    for (pos1, quat1), (pos2, quat2) in zip(ee_path[:-1], ee_path[1:]):
        linear_distance = np.linalg.norm(np.array(pos2) - np.array(pos1))
        angular_distance = angle_distance(quat1, quat2)
        nr_steps = max(int(max(linear_distance / max_linear_velocity, angular_distance / max_angular_velocity)), 1)

        # If the angular distance is too small, we can just use linear interpolation.
        if angular_distance < 1e-3:
            for i in range(1, nr_steps + 1):
                smooth_ee_path.append((pos1 + (pos2 - pos1) * i / nr_steps, normalize_vector(quat1 + (quat2 - quat1) * i / nr_steps)))
        else:
            for i in range(1, nr_steps + 1):
                smooth_ee_path.append((pos1, slerp(quat1, quat2, i / nr_steps)))

    return smooth_ee_path
