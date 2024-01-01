#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : path_finding.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/08/2022
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

from typing import Tuple, NamedTuple
from concepts.algorithm.search.heuristic_search import run_heuristic_search
from concepts.benchmark.gridworld.minigrid.gym_minigrid.minigrid import MiniGridEnv, DIR_TO_VEC

__all__ = ['find_path', 'find_path_to_obj']


class _PathFindingState(NamedTuple):
    pose: Tuple[int, int]
    dir: int


NAVIGATION_ACTIONS = [MiniGridEnv.Actions.forward, MiniGridEnv.Actions.left, MiniGridEnv.Actions.right]


def gen_get_navigation_successors(env: MiniGridEnv):
    def get_successor(state: _PathFindingState):
        for action in NAVIGATION_ACTIONS:
            if action is MiniGridEnv.Actions.forward:
                dx, dy = DIR_TO_VEC[state.dir]
                xx, yy = state.pose[0] + dx, state.pose[1] + dy
                cell = env.grid.get(xx, yy)
                if cell is None or cell.can_overlap():
                    yield action, _PathFindingState((xx, yy), state.dir), 1
            elif action is MiniGridEnv.Actions.left:
                yield action, _PathFindingState(
                    state.pose, ((state.dir - 1) + 4) % 4
                    ), 1
            elif action is MiniGridEnv.Actions.right:
                yield action, _PathFindingState(state.pose, (state.dir + 1) % 4), 1
            else:
                raise ValueError('Unknown action: {}.'.format(action))
    return get_successor


def find_path(env: MiniGridEnv, target_pose: Tuple[int, int], target_dir: int):
    target_pose = tuple(target_pose)
    target_dir = int(target_dir)

    def check_goal(state: _PathFindingState):
        pose, dir = state
        return pose == target_pose and dir == target_dir

    def get_priority(state: _PathFindingState, g: int):
        return abs(state.pose[0] - target_pose[0]) + abs(state.pose[1] - target_pose[1]) + int(target_dir != state.dir) + g

    state = _PathFindingState(env.agent_pos, env.agent_dir)
    try:
        return run_heuristic_search(
            state,
            check_goal,
            get_priority,
            gen_get_navigation_successors(env),
        )[1]
    except RuntimeError:
        return None


def find_path_to_obj(env: MiniGridEnv, target_obj_pos: Tuple[int, int]):
    target_obj_pose = tuple(target_obj_pos)

    def check_goal(state: _PathFindingState):
        dx, dy = DIR_TO_VEC[state.dir]
        xx, yy = state.pose[0] + dx, state.pose[1] + dy
        return (xx, yy) == target_obj_pose

    def get_priority(state: _PathFindingState, g: int):
        return abs(state.pose[0] - target_obj_pose[0]) + abs(state.pose[1] - target_obj_pose[1]) - 1 + g

    init_state = _PathFindingState(env.agent_pos, env.agent_dir)
    try:
        return run_heuristic_search(
            init_state,
            check_goal,
            get_priority,
            gen_get_navigation_successors(env),
        )[1]
    except RuntimeError:
        return None
