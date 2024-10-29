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

from concepts.simulator.pybullet.components.panda.panda_robot import PandaRobot
from concepts.utils.typing_utils import Vec3f, Vec4f, VecNf

__all__ = [
    'gen_synchronized_collision_free_qpos_path_from_current_qpos_and_ee_path',
    'gen_synchronized_collision_free_qpos_path_from_current_qpos_and_ee_linear_path',
    'gen_synchronized_smooth_qpos_path_from_qpos_path'
]


def gen_synchronized_collision_free_qpos_path_from_current_qpos_and_ee_path(
    robot1: PandaRobot, robot1_waypoints: Sequence[Tuple[Vec3f, Vec4f]],
    robot2: PandaRobot, robot2_waypoints: Sequence[Tuple[Vec3f, Vec4f]],
    *,
    first_qpos1: Optional[VecNf] = None, first_qpos2: Optional[VecNf] = None,
    nr_ik_attempts: int = 1,
    ignored_collision_bodies: Optional[List[int]] = None,
    collision_free_planning: bool = True,
    max_joint_distance_between_waypoints: float = float('inf'),
    generator: bool = False,
    verbose: bool = False,
) -> Union[Iterator[Tuple[List[VecNf], List[VecNf]]], Tuple[Optional[List[VecNf]], Optional[List[VecNf]]]]:
    assert len(robot1_waypoints) == len(robot2_waypoints)

    def dfs(remaining_waypoints1, remaining_waypoints2, current_qpos1: VecNf, current_qpos2: VecNf) -> Iterator[Tuple[List[VecNf], List[VecNf]]]:
        if len(remaining_waypoints1) == 0:
            yield list(), list()
            return
        target_pos1, target_quat1 = remaining_waypoints1[0]
        target_pos2, target_quat2 = remaining_waypoints2[0]
        for _ in range(nr_ik_attempts):
            if len(remaining_waypoints1) == 1 and len(remaining_waypoints2) == 1 and first_qpos1 is not None and first_qpos2 is not None:
                solution1 = first_qpos1
                solution2 = first_qpos2
            else:
                solution1 = robot1.ikfast(target_pos1, target_quat1, last_qpos=current_qpos1, error_on_fail=False, max_distance=max_joint_distance_between_waypoints)
                solution2 = robot2.ikfast(target_pos2, target_quat2, last_qpos=current_qpos2, error_on_fail=False, max_distance=max_joint_distance_between_waypoints)

                if solution1 is None or solution2 is None:
                    continue
                if robot1.is_colliding_with_saved_state(solution1, ignored_collision_bodies=ignored_collision_bodies):
                    continue
                if robot2.is_colliding_with_saved_state(solution2, ignored_collision_bodies=ignored_collision_bodies):
                    continue

            if collision_free_planning:
                rv1, path1 = robot1.rrt_collision_free(current_qpos1, solution1, smooth_fine_path=True, ignored_collision_bodies=ignored_collision_bodies, disable_renderer=True, verbose=verbose)
                rv2, path2 = robot2.rrt_collision_free(current_qpos2, solution2, smooth_fine_path=True, ignored_collision_bodies=ignored_collision_bodies, disable_renderer=True, verbose=verbose)
                if rv1 and rv2:
                    for remaining_path1, remaining_path2 in dfs(remaining_waypoints1[1:], remaining_waypoints2[1:], path1[-1], path2[-1]):
                        yield path1 + remaining_path1, path2 + remaining_path2
            else:
                for remaining_path1, remaining_path2 in dfs(remaining_waypoints1[1:], remaining_waypoints2[1:], solution1, solution2):
                    yield [solution1] + remaining_path1, [solution2] + remaining_path2

    if generator:
        return dfs(robot1_waypoints, robot2_waypoints, robot1.get_qpos(), robot2.get_qpos())

    for x, y in dfs(robot1_waypoints, robot2_waypoints, robot1.get_qpos(), robot2.get_qpos()):
        return x, y
    return None, None


def gen_synchronized_collision_free_qpos_path_from_current_qpos_and_ee_linear_path(
    robot1: PandaRobot, robot1_target_pos1: Vec3f, robot1_target_pos2: Vec3f,
    robot2: PandaRobot, robot2_target_pos1: Vec3f, robot2_target_pos2: Vec3f,
    *,
    robot1_ee_dir: Optional[Vec3f] = (0, 0, -1), robot1_ee_dir2: Optional[Vec3f] = (1, 0, 0),
    robot2_ee_dir: Optional[Vec3f] = (0, 0, -1), robot2_ee_dir2: Optional[Vec3f] = (1, 0, 0),
    nr_waypoints: int = 10, ignored_collision_bodies: Optional[List[int]] = None, return_smooth_path: bool = False,
    max_joint_distance_between_waypoints: float = float('inf'),
    nr_ik_attempts: int = 1, collision_free_planning: bool = True
):
    robot1_quat = robot1.get_ee_quat_from_vectors(robot1_ee_dir, robot1_ee_dir2)
    robot2_quat = robot2.get_ee_quat_from_vectors(robot2_ee_dir, robot2_ee_dir2)

    robot1_steps = np.linspace(robot1_target_pos1, robot1_target_pos2, nr_waypoints)
    robot2_steps = np.linspace(robot2_target_pos1, robot2_target_pos2, nr_waypoints)
    robot1_waypoints = [(step, robot1_quat) for step in robot1_steps]
    robot2_waypoints = [(step, robot2_quat) for step in robot2_steps]

    path1, path2 = gen_synchronized_collision_free_qpos_path_from_current_qpos_and_ee_path(
        robot1, robot1_waypoints, robot2, robot2_waypoints, ignored_collision_bodies=ignored_collision_bodies, nr_ik_attempts=nr_ik_attempts, collision_free_planning=collision_free_planning,
        max_joint_distance_between_waypoints=max_joint_distance_between_waypoints
    )
    return (path1, path2) if not return_smooth_path else gen_synchronized_smooth_qpos_path_from_qpos_path(robot1, robot2, path1, path2)


def gen_synchronized_smooth_qpos_path_from_qpos_path(robot1: PandaRobot, robot2: PandaRobot, qt1: Optional[List[VecNf]], qt2: Optional[List[VecNf]]):
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
