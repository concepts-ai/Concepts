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

from concepts.math.frame_utils_xyzw import get_transform_a_to_b
from concepts.simulator.pybullet.components.robot_base import BulletArmRobotBase
from concepts.utils.typing_utils import Vec3f, Vec4f, Vec7f

__all__ = [
    'gen_qpos_path_from_current_qpos_and_target_qpos',
    'calc_grasp_pose_from_current_object_pose_and_ee_pose',
    'gen_qpos_path_from_ee_path', 'gen_qpos_path_from_ee_path_pybullet',
    'is_collision_free_qpos', 'is_collision_free_ee_pose', 'is_collision_free_qpos_path',
    'gen_collision_free_qpos_path_from_current_qpos_and_ee_path', 'gen_collision_free_qpos_path_from_current_qpos_and_ee_linear_path',
    'gen_qpos_path_from_current_qpos_and_ee_pose', 'gen_collision_free_path_from_current_qpos_and_ee_pose',
    'gen_smooth_qpos_path_from_qpos_path', 'smooth_move_qpos_trajectory'
]


def gen_qpos_path_from_current_qpos_and_target_qpos(robot: BulletArmRobotBase, target_qpos: np.ndarray, ignored_collision_bodies: Optional[List[int]] = None, verbose: bool = False) -> Optional[List[np.ndarray]]:
    """Generate a collision-free path from the current qpos to reach the target qpos.

    Args:
        robot: the robot.
        target_qpos: the target qpos.
        ignored_collision_bodies: the object ids to ignore in collision checking.
        verbose: whether to print additional information.

    Returns:
        the qpos path if successful.
    """
    rv, path = robot.rrt_collision_free(robot.get_qpos(), target_qpos, smooth_fine_path=True, ignored_collision_bodies=ignored_collision_bodies, disable_renderer=True, verbose=verbose)
    if not rv:
        return None
    return path


def calc_grasp_pose_from_current_object_pose_and_ee_pose(robot: BulletArmRobotBase, object_id: int, ee_pos: Vec3f, ee_dir: Vec3f = (0., 0., -1.), ee_dir2: Vec3f = (1., 0., 0.)) -> Tuple[Vec3f, Vec4f]:
    """Compute the grasp pose from the current object pose and the desired ee pose.

    Args:
        robot: the robot.
        object_id: the object id.
        ee_pos: the desired ee position.
        ee_dir: the desired ee direction.
        ee_dir2: the desired ee direction 2.
    """

    ee_quat = robot.get_ee_quat_from_vectors(ee_dir, ee_dir2)
    object_state = robot.world.get_body_state_by_id(object_id)
    ee_to_object = get_transform_a_to_b(ee_pos, ee_quat, object_state.pos, object_state.quat)
    return ee_to_object


def gen_qpos_path_from_ee_path(robot: BulletArmRobotBase, trajectory: List[Tuple[np.ndarray, np.ndarray]], max_pairwise_distance: float = 0.1, verbose: bool = False) -> Tuple[bool, List[np.ndarray]]:
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
    last_qpos = robot.ikfast(trajectory[0][0], trajectory[0][1], error_on_fail=False)
    if last_qpos is None:
        if verbose:
            print('qpos_trajectory_from_poses: failed to compute the first qpos.')
        return False, qpos_trajectory
    qpos_trajectory.append(last_qpos)

    success = True
    for i, (this_pos, this_quat) in enumerate(trajectory):
        qpos = robot.ikfast(this_pos, this_quat, max_distance=max_pairwise_distance, last_qpos=last_qpos, error_on_fail=False)
        if qpos is None:
            if verbose:
                print(f'qpos_trajectory_from_poses: failed to compute the qpos at step #{i+1}.')
            return False, qpos_trajectory
        qpos_trajectory.append(qpos)
        last_qpos = qpos

    return success, qpos_trajectory


def gen_qpos_path_from_ee_path_pybullet(robot: BulletArmRobotBase, trajectory: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[bool, List[np.ndarray]]:
    qpos_trajectory = list()
    for pos, quat in trajectory:
        qpos = robot.ik_pybullet(pos, quat)
        if qpos is None:
            return False, qpos_trajectory
        qpos_trajectory.append(qpos)
    return True, qpos_trajectory


def is_collision_free_qpos(robot: BulletArmRobotBase, qpos: np.ndarray, exclude: Optional[List[int]] = None, verbose: bool = False) -> bool:
    """Check whether the given qpos is collision free. The function also accepts a list of object ids to exclude (e.g., the object in hand).

    Args:
        robot: the robot.
        qpos: the qpos to check.
        exclude: the object ids to exclude.
        verbose: whether to print the collision information.

    Returns:
        True if the qpos is collision free.
    """
    with robot.world.save_body(robot.get_body_id()):
        robot.set_qpos(qpos)
        contacts = robot.world.get_contact(robot.get_body_id(), update=True)
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


def is_collision_free_ee_pose(robot: BulletArmRobotBase, pos: np.ndarray, quat: Optional[np.ndarray] = None, exclude: Optional[List[int]] = None, verbose: bool = False) -> bool:
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
    return is_collision_free_qpos(robot, qpos, exclude=exclude, verbose=verbose)


def is_collision_free_qpos_path(robot: BulletArmRobotBase, qpos_trajectory: List[np.ndarray], exclude: Optional[List[int]] = None, verbose: bool = False) -> bool:
    for qpos in qpos_trajectory:
        if not is_collision_free_qpos(robot, qpos, exclude=exclude, verbose=verbose):
            return False
    return True


def gen_collision_free_qpos_path_from_current_qpos_and_ee_path(
    robot: BulletArmRobotBase, waypoints: Sequence[Tuple[Vec3f, Vec4f]], nr_ik_attempts: int = 1,
    ignored_collision_bodies: Optional[List[int]] = None,
    collision_free_planning: bool = True,
    generator: bool = False,
    verbose: bool = False,
) -> Union[Iterator[List[Vec7f]], Optional[List[Vec7f]]]:
    def dfs(remaining_waypoints, current_qpos: Vec7f) -> Iterator[List[Vec7f]]:
        if len(remaining_waypoints) == 0:
            yield list()
            return
        target_pos, target_quat = remaining_waypoints[0]
        for _ in range(nr_ik_attempts):
            solution = robot.ikfast(target_pos, target_quat, last_qpos=current_qpos, error_on_fail=False)
            if verbose:
                print(f'Generating IK (remaining waypoints: {len(remaining_waypoints)}): {solution is not None}')
            if solution is None:
                continue
            if robot.is_colliding_with_saved_state(solution, ignored_collision_bodies=ignored_collision_bodies):
                if verbose:
                    print(f'Collision detected (remaining waypoints: {len(remaining_waypoints)}): {solution is not None}')
                    with robot.world.save_world_builtin():
                        robot.set_qpos_with_attached_objects(solution)
                        collisions = robot.is_colliding(return_contacts=True, ignored_collision_bodies=ignored_collision_bodies)
                        print([[collision.body_a_name, collision.body_b_name] for collision in collisions])
                        input('Press any key to continue')
                continue
            if collision_free_planning:
                rv, path = robot.rrt_collision_free(current_qpos, solution, smooth_fine_path=True, ignored_collision_bodies=ignored_collision_bodies, disable_renderer=True, verbose=verbose)
                if verbose:
                    print(f'Generating path (remaining waypoints: {len(remaining_waypoints)}): {rv} {len(path) if path is not None else 0}')
                if rv:
                    for remaining_path in dfs(remaining_waypoints[1:], path[-1]):
                        yield path + remaining_path
            else:
                for remaining_path in dfs(remaining_waypoints[1:], solution):
                    yield [solution] + remaining_path

    if generator:
        return dfs(waypoints, robot.get_qpos())

    for x in dfs(waypoints, robot.get_qpos()):
        return x
    return None


def gen_collision_free_qpos_path_from_current_qpos_and_ee_linear_path(
    robot: BulletArmRobotBase, target_pos1, target_pos2, ee_dir: Optional[Vec3f] = (0, 0, -1), ee_dir2: Optional[Vec3f] = (1, 0, 0), nr_waypoints: int = 10,
    ignored_collision_bodies: Optional[List[int]] = None, return_smooth_path: bool = False,
    *,
    nr_ik_attempts: int = 1, collision_free_planning: bool = True
):
    """Generate a collision-free Cartesian path from the current qpos to follow a straight line from target_pos1 to target_pos2."""
    quat = robot.get_ee_quat_from_vectors(ee_dir, ee_dir2)
    steps = np.linspace(target_pos1, target_pos2, nr_waypoints)
    waypoints = [(step, quat) for step in steps]

    path = gen_collision_free_qpos_path_from_current_qpos_and_ee_path(robot, waypoints, ignored_collision_bodies=ignored_collision_bodies, nr_ik_attempts=nr_ik_attempts, collision_free_planning=collision_free_planning)
    return path if not return_smooth_path else gen_smooth_qpos_path_from_qpos_path(robot, path)


def gen_qpos_path_from_current_qpos_and_ee_pose(
    robot: BulletArmRobotBase, target_pos: Vec3f, ee_dir: Optional[Vec3f] = (0, 0, -1), ee_dir2: Optional[Vec3f] = (1, 0, 0),
    return_smooth_path: bool = False, *,
    nr_ik_attempts: int = 1
):
    """Generate a collision-free path from the current qpos to reach the target_pos and target_quat."""
    target_quat = robot.get_ee_quat_from_vectors(ee_dir, ee_dir2)
    waypoints = [(target_pos, target_quat)]

    path = gen_collision_free_qpos_path_from_current_qpos_and_ee_path(robot, waypoints, nr_ik_attempts=nr_ik_attempts, collision_free_planning=False)
    return path if not return_smooth_path else gen_smooth_qpos_path_from_qpos_path(robot, path)


def gen_collision_free_path_from_current_qpos_and_ee_pose(
    robot: BulletArmRobotBase, target_pos: Vec3f, ee_dir: Optional[Vec3f] = (0, 0, -1), ee_dir2: Optional[Vec3f] = (1, 0, 0),
    ignored_collision_bodies: Optional[List[int]] = None, return_smooth_path: bool = False, *,
    nr_ik_attempts: int = 1,
    visualize: bool = False,
    verbose: bool = False
):
    """Generate a collision-free path from the current qpos to reach the target_pos."""
    quat = robot.get_ee_quat_from_vectors(ee_dir, ee_dir2)
    waypoints = [(target_pos, quat)]

    path = gen_collision_free_qpos_path_from_current_qpos_and_ee_path(robot, waypoints, ignored_collision_bodies=ignored_collision_bodies, nr_ik_attempts=nr_ik_attempts, verbose=verbose)
    if visualize:
        print('-' * 80)
        print(f'Visualizing collision free path length: {len(path) if path is not None else "None"}')
        if path is not None:
            robot.replay_qpos_trajectory(path)
        print('Visualizing collision free path finished')
        print()
    return path if not return_smooth_path else gen_smooth_qpos_path_from_qpos_path(robot, path)


def gen_smooth_qpos_path_from_qpos_path(robot: BulletArmRobotBase, qt: Optional[List[Vec7f]]):
    if qt is None:
        return None

    qt = qt.copy()
    qt.insert(0, robot.get_qpos())

    smooth_qt = [qt[0]]
    cspace = robot.get_configuration_space()
    for qpos1, qpos2 in zip(qt[:-1], qt[1:]):
        smooth_qt.extend(cspace.gen_path(qpos1, qpos2)[1][1:])
    return smooth_qt


def smooth_move_qpos_trajectory(robot: BulletArmRobotBase, qt: List[Vec7f], **kwargs):
    smooth_qt = gen_smooth_qpos_path_from_qpos_path(robot, qt)
    return robot.move_qpos_path_v2(smooth_qt, timeout=kwargs.get('timeout', 10))

