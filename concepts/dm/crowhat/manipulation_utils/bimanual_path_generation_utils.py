#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : bimanual_path_generation_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/01/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
from typing import Optional, Union, Iterator, Sequence, Tuple, List

from concepts.math.rotationlib_xyzw import slerp, normalize_vector
from concepts.dm.crowhat.world.manipulator_interface import SingleGroupMotionPlanningInterface
from concepts.dm.crowhat.manipulation_utils.pose_utils import angle_distance
from concepts.utils.typing_utils import Vec3f, Vec4f, VecNf

__all__ = [
    'gen_synchronized_collision_free_qpos_path_from_current_qpos_and_ee_path',
    'gen_synchronized_collision_free_qpos_path_from_current_qpos_and_ee_linear_path',
    'calc_synchronized_smooth_qpos_path_from_qpos_path',
    'wrap_iterator_sync_smooth_qpos_path',
    'calc_synchronized_smooth_ee_path_from_ee_path'
]


def gen_synchronized_collision_free_qpos_path_from_current_qpos_and_ee_path(
    robot1: SingleGroupMotionPlanningInterface, robot1_waypoints: Sequence[Tuple[Vec3f, Vec4f]],
    robot2: SingleGroupMotionPlanningInterface, robot2_waypoints: Sequence[Tuple[Vec3f, Vec4f]],
    *,
    first_qpos1: Optional[VecNf] = None, first_qpos2: Optional[VecNf] = None,
    nr_ik_attempts: int = 1, max_joint_distance_between_waypoints: float = float('inf'),
    check_collision: bool = True, collision_free_planning: bool = True, ignored_collision_bodies: Optional[List[int]] = None,
    return_smooth_path: bool = False, generator: bool = False, verbose: bool = False,
) -> Union[Iterator[Tuple[List[VecNf], List[VecNf]]], Tuple[Optional[List[VecNf]], Optional[List[VecNf]]]]:
    assert len(robot1_waypoints) == len(robot2_waypoints)

    if collision_free_planning:
        raise NotImplementedError('Collision free planning is not supported yet.')

    def dfs(remaining_waypoints1, remaining_waypoints2, current_qpos1: VecNf, current_qpos2: VecNf, is_first: bool = False) -> Iterator[Tuple[List[VecNf], List[VecNf]]]:
        if len(remaining_waypoints1) == 0:
            yield list(), list()
            return
        target_pos1, target_quat1 = remaining_waypoints1[0]
        target_pos2, target_quat2 = remaining_waypoints2[0]
        for _ in range(nr_ik_attempts):
            if is_first and first_qpos1 is not None and first_qpos2 is not None:
                solution1 = first_qpos1
                solution2 = first_qpos2
            else:
                solution1 = robot1.ik(target_pos1, target_quat1, qpos=current_qpos1, max_distance=max_joint_distance_between_waypoints)
                solution2 = robot2.ik(target_pos2, target_quat2, qpos=current_qpos2, max_distance=max_joint_distance_between_waypoints)

                if verbose:
                    print('IK:', solution1, solution2)

                if solution1 is None or solution2 is None:
                    if solution1 is None:
                        if verbose:
                            print('IK failed for robot 1: desired pose', target_pos1, target_quat1)
                            print('Previous qpos:', current_qpos1, 'with previous pose:', robot1.fk(current_qpos1))
                    if solution2 is None:
                        if verbose:
                            print('IK failed for robot 2: desired pose', target_pos2, target_quat2)
                            print('Previous qpos:', current_qpos2, 'with previous pose:', robot2.fk(current_qpos2))
                    continue

                if check_collision:
                    if collision_list := robot1.check_collision(solution1, ignored_collision_bodies=ignored_collision_bodies, return_list=True):
                        if verbose:
                            print('Collision detected for robot 1: desired pose', target_pos1, target_quat1)
                            print('Collision list:', [robot1.planning_world.get_object_name(x) for x in collision_list])
                        continue
                    if collision_list := robot2.check_collision(solution2, ignored_collision_bodies=ignored_collision_bodies, return_list=True):
                        if verbose:
                            print('Collision detected for robot 2: desired pose', target_pos2, target_quat2)
                            print('Collision list:', [robot2.planning_world.get_object_name(x) for x in collision_list])
                        continue

            if collision_free_planning:
                rv1, path1 = robot1.rrt_collision_free(current_qpos1, solution1, smooth_fine_path=True, ignored_collision_bodies=ignored_collision_bodies, verbose=verbose)
                rv2, path2 = robot2.rrt_collision_free(current_qpos2, solution2, smooth_fine_path=True, ignored_collision_bodies=ignored_collision_bodies, verbose=verbose)
                if rv1 and rv2:
                    for remaining_path1, remaining_path2 in dfs(remaining_waypoints1[1:], remaining_waypoints2[1:], path1[-1], path2[-1]):
                        yield path1 + remaining_path1, path2 + remaining_path2
            else:
                for remaining_path1, remaining_path2 in dfs(remaining_waypoints1[1:], remaining_waypoints2[1:], solution1, solution2):
                    yield [solution1] + remaining_path1, [solution2] + remaining_path2

    if generator:
        if return_smooth_path:
            return wrap_iterator_sync_smooth_qpos_path(robot1, robot2, dfs(robot1_waypoints, robot2_waypoints, robot1.get_qpos(), robot2.get_qpos(), is_first=True))
        else:
            return dfs(robot1_waypoints, robot2_waypoints, robot1.get_qpos(), robot2.get_qpos(), is_first=True)

    for rv in dfs(robot1_waypoints, robot2_waypoints, robot1.get_qpos(), robot2.get_qpos(), is_first=True):
        if return_smooth_path:
            return calc_synchronized_smooth_qpos_path_from_qpos_path(robot1, robot2, rv[0], rv[1])
        return rv
    return None, None


def gen_synchronized_collision_free_qpos_path_from_current_qpos_and_ee_linear_path(
    robot1: SingleGroupMotionPlanningInterface, robot1_target_pos1: Vec3f, robot1_target_pos2: Vec3f,
    robot2: SingleGroupMotionPlanningInterface, robot2_target_pos1: Vec3f, robot2_target_pos2: Vec3f,
    *,
    first_qpos1: Optional[VecNf] = None, first_qpos2: Optional[VecNf] = None,
    robot1_ee_dir: Optional[Vec3f] = (0, 0, -1), robot1_ee_dir2: Optional[Vec3f] = (1, 0, 0),
    robot2_ee_dir: Optional[Vec3f] = (0, 0, -1), robot2_ee_dir2: Optional[Vec3f] = (1, 0, 0),
    nr_waypoints: int = 10,
    nr_ik_attempts: int = 1, max_joint_distance_between_waypoints: float = float('inf'),
    check_collision: bool = True, collision_free_planning: bool = True, ignored_collision_bodies: Optional[List[int]] = None,
    return_smooth_path: bool = False, generator: bool = False, verbose: bool = False,
) -> Iterator[Tuple[List[VecNf], List[VecNf]]]:
    robot1_quat = robot1.calc_ee_quat_from_vectors(robot1_ee_dir, robot1_ee_dir2)
    robot2_quat = robot2.calc_ee_quat_from_vectors(robot2_ee_dir, robot2_ee_dir2)

    robot1_steps = np.linspace(robot1_target_pos1, robot1_target_pos2, nr_waypoints)
    robot2_steps = np.linspace(robot2_target_pos1, robot2_target_pos2, nr_waypoints)
    robot1_waypoints = [(step, robot1_quat) for step in robot1_steps]
    robot2_waypoints = [(step, robot2_quat) for step in robot2_steps]

    return gen_synchronized_collision_free_qpos_path_from_current_qpos_and_ee_path(
        robot1, robot1_waypoints, robot2, robot2_waypoints,
        first_qpos1=first_qpos1, first_qpos2=first_qpos2,
        ignored_collision_bodies=ignored_collision_bodies,
        nr_ik_attempts=nr_ik_attempts, check_collision=check_collision, collision_free_planning=collision_free_planning,
        max_joint_distance_between_waypoints=max_joint_distance_between_waypoints,
        return_smooth_path=return_smooth_path,
        generator=generator, verbose=verbose
    )


def calc_synchronized_smooth_qpos_path_from_qpos_path(robot1: SingleGroupMotionPlanningInterface, robot2: SingleGroupMotionPlanningInterface, qt1: Optional[List[VecNf]], qt2: Optional[List[VecNf]]):
    if qt1 is None or qt2 is None:
        return None, None

    assert len(qt1) == len(qt2)

    qt1 = qt1.copy()
    qt2 = qt2.copy()
    qt1.insert(0, robot1.get_qpos())
    qt2.insert(0, robot2.get_qpos())

    smooth_qt1 = [qt1[0]]
    smooth_qt2 = [qt2[0]]
    cspace1 = robot1.get_configuration_space()
    cspace2 = robot2.get_configuration_space()

    for robot1_qpos1, robot1_qpos2, robot2_qpos1, robot2_qpos2 in zip(qt1[:-1], qt1[1:], qt2[:-1], qt2[1:]):
        subpath1 = cspace1.gen_path(robot1_qpos1, robot1_qpos2)[1][1:]
        subpath2 = cspace2.gen_path(robot2_qpos1, robot2_qpos2)[1][1:]

        max_len = max(len(subpath1), len(subpath2))
        smooth_qt1.extend(cspace1.gen_interpolated_path(robot1_qpos1, robot1_qpos2, max_len)[1:])
        smooth_qt2.extend(cspace2.gen_interpolated_path(robot2_qpos1, robot2_qpos2, max_len)[1:])

    assert len(smooth_qt1) == len(smooth_qt2)
    return smooth_qt1, smooth_qt2


def wrap_iterator_sync_smooth_qpos_path(robot1: SingleGroupMotionPlanningInterface, robot2: SingleGroupMotionPlanningInterface, qts_iterator: Iterator[Tuple[List[VecNf], List[VecNf]]]) -> Iterator[Tuple[List[VecNf], List[VecNf]]]:
    for qts in qts_iterator:
        yield calc_synchronized_smooth_qpos_path_from_qpos_path(robot1, robot2, qts[0], qts[1])


def calc_synchronized_smooth_ee_path_from_ee_path(
    ee_path1: Optional[List[Tuple[Vec3f, Vec4f]]], ee_path2: Optional[List[Tuple[Vec3f, Vec4f]]],
    max_linear_velocity: float = 0.1, max_angular_velocity: float = np.pi / 4, fps: int = 60
) -> Tuple[Optional[List[Tuple[Vec3f, Vec4f]]], Optional[List[Tuple[Vec3f, Vec4f]]]]:
    max_linear_velocity = max_linear_velocity / fps
    max_angular_velocity = max_angular_velocity / fps

    if ee_path1 is None or ee_path2 is None:
        return None, None

    assert len(ee_path1) == len(ee_path2)

    smooth_ee_path1 = [ee_path1[0]]
    smooth_ee_path2 = [ee_path2[0]]

    for (posa1, quata1), (posa2, quata2), (posb1, quatb1), (posb2, quatb2) in zip(ee_path1[:-1], ee_path1[1:], ee_path2[:-1], ee_path2[1:]):
        linear_distance1 = np.linalg.norm(np.array(posa2) - np.array(posa1))
        angular_distance1 = angle_distance(quata1, quata2)
        linear_distance2 = np.linalg.norm(np.array(posb2) - np.array(posb1))
        angular_distance2 = angle_distance(quatb1, quatb2)
        nr_steps = max(int(max(
            linear_distance1 / max_linear_velocity, angular_distance1 / max_angular_velocity,
            linear_distance2 / max_linear_velocity, angular_distance2 / max_angular_velocity
        )), 1)

        print('Smooth path:', linear_distance1, angular_distance1, linear_distance2, angular_distance2, nr_steps)

        if angular_distance1 < 1e-3:
            for i in range(1, nr_steps + 1):
                smooth_ee_path1.append((posa1 + (posa2 - posa1) * i / nr_steps, normalize_vector(quata1 + (quata2 - quata1) * i / nr_steps)))
        else:
            for i in range(1, nr_steps + 1):
                smooth_ee_path1.append((posa1, slerp(quata1, quata2, i / nr_steps)))

        if angular_distance2 < 1e-3:
            for i in range(1, nr_steps + 1):
                smooth_ee_path2.append((posb1 + (posb2 - posb1) * i / nr_steps, normalize_vector(quatb1 + (quatb2 - quatb1) * i / nr_steps)))
        else:
            for i in range(1, nr_steps + 1):
                smooth_ee_path2.append((posb1, slerp(quatb1, quatb2, i / nr_steps)))

    return smooth_ee_path1, smooth_ee_path2
