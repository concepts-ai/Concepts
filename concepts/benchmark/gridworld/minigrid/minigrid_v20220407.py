#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : minigrid_v20220407.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/07/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp
import time
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np
import gym.spaces as spaces
import torch
import jactorch
# import hacl.pdsketch as pds
# import hacl.pdsketch.rl as pdsrl
# import hacl.envs.gridworld.minigrid.gym_minigrid as minigrid
# from .gym_minigrid.path_finding import find_path_to_obj

import concepts.benchmark.gridworld.minigrid.gym_minigrid as minigrid
from concepts.benchmark.gridworld.minigrid.gym_minigrid.minigrid import MiniGridEnv
from concepts.benchmark.gridworld.minigrid.gym_minigrid.path_finding import find_path_to_obj

from concepts.dm.pdsketch.domain import State, Domain
from concepts.dm.pdsketch.executor import PDSketchExecutor

__all__ = [
    'MiniGridEnvV20220407', 'make_minigrid_env', 'get_minigrid_domain_filename',
    'visualize_minigrid_planner', 'visualize_minigrid_plan'
]


def _map_int(x):
    if isinstance(x, tuple):
        return map(int, x)
    if isinstance(x, np.ndarray):
        return map(int, x)
    if isinstance(x, torch.Tensor):
        return map(int, jactorch.as_numpy(x))


@dataclass
class MiniGridEnvAction(object):
    name: str
    arguments: Tuple[int, ...] = tuple()


class MiniGridEnvV20220407(MiniGridEnv):
    SUPPORTED_TASKS = ['gotosingle', 'goto', 'goto2', 'pickup', 'open', 'generalization']
    SUPPORTED_ENCODING = ['full']

    def __init__(self, task='pickup', encoding: str = 'full'):
        assert task in type(self).SUPPORTED_TASKS, f'Unknown task: {task}.'
        assert encoding in type(self).SUPPORTED_ENCODING, f'Unknown encoding: {encoding}.'

        self.task = task
        self.encoding = encoding
        self.encoding_executor = None
        self.options = dict()

        self.goal_obj: Optional[minigrid.WorldObj] = None
        self.goal_pose: Optional[Tuple[int, int]] = None
        self.mission: str = ''

        super().__init__(grid_size=7, max_steps=64, seed=1337, require_obs=False)

    action_space: spaces.Discrete
    observation_space: spaces.Box

    task: str
    """A short string describing the task."""

    encoding: str
    """A short string describing the encoding method."""

    encoding_executor: Optional[PDSketchExecutor]
    """The :class:`~concepts.dm.pdsketch.executor.PDSketchExecutor` used for encoding the states."""

    goal_obj: Optional[minigrid.WorldObj]
    """The goal object."""

    goal_pose: Optional[Tuple[int, int]]
    """The goal pose."""

    mission: str
    """The mission string."""

    def set_options(self, **kwargs):
        self.options.update(kwargs)

    def get_option(self, name, default=None):
        return self.options.get(name, default)

    def set_encoding_executor(self, executor: PDSketchExecutor):
        self.encoding_executor = executor

    @property
    def encoding_domain(self) -> Domain:
        return self.encoding_executor.domain

    def _gen_grid(self, width, height):
        if self.task == 'gotosingle':
            _gen_grid_goto_single(self, width, height)
        elif self.task in ('goto', 'goto2'):
            _gen_grid_goto(self, width, height)
        elif self.task == 'pickup':
            _gen_grid_pickup(self, width, height)
        elif self.task == 'open':
            _gen_grid_open(self, width, height)
        elif self.task == 'generalization':
            _gen_grid_generalization(self, width, height)
        else:
            raise ValueError(f'Unknown task: {self.task}.')

    def reset(self):
        super().reset()
        return self.compute_obs()

    def step(self, action: MiniGridEnvAction):
        if action.name == 'move':
            self.step_move_to(action.arguments[0], action.arguments[1])
        elif action.name == 'forward':
            self.step_forward()
        elif action.name == 'lturn':
            self.step_lturn()
        elif action.name == 'rturn':
            self.step_rturn()
        elif action.name == 'pickup':
            self.step_pickup()
        elif action.name == 'place':
            self.step_drop()
        elif action.name == 'toggle':
            self.step_inner(self.Actions.toggle)
        else:
            raise ValueError(f'Unknown action: {action}.')

        obs = self.compute_obs()
        done = self.compute_done()
        return obs, -1, done, {}

    def compute_obs(self):
        state = self.get_pds_state()
        return {'state': state, 'mission': self.mission}

    def compute_done(self):
        if self.task in ('goto', 'goto2', 'gotosingle'):
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell is not None and fwd_cell.type == self.goal_obj.type and fwd_cell.color == self.goal_obj.color:
                return True
        elif self.task == 'pickup':
            if self.carrying is not None and self.carrying.color == self.goal_obj.color and self.carrying.type == self.goal_obj.type:
                return True
        elif self.task == 'open':
            for _, _, obj in self.iter_objects():
                if obj.color == self.goal_obj.color and obj.type == self.goal_obj.type and obj.is_open:
                    return True
        elif self.task == 'generalization':
            if self.carrying is not None and self.carrying.color == self.goal_obj[0].color and self.carrying.type == self.goal_obj[0].type:
                fwd_pos = self.front_pos
                fwd_cell = self.grid.get(*fwd_pos)
                if fwd_cell is not None and fwd_cell.type == self.goal_obj[1].type and fwd_cell.color == self.goal_obj[1].color:
                    return True
        else:
            raise ValueError(f'Unknown task: {self.task}.')
        return False

    def step_inner(self, action):
        super().step(action)

    def step_move_to(self, pose, dir, traj=None):
        x, y = _map_int(pose)
        dir, = _map_int(dir)
        if self.grid.get(x, y) is None or self.grid.get(x, y).can_overlap():
            self.agent_pos = (x, y)
            self.agent_dir = dir

    def step_pickup(self):
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        if fwd_cell and fwd_cell.can_pickup():
            if self.carrying is None:
                self.carrying = fwd_cell
                self.carrying.cur_pos = (-1, -1)
                self.grid.set(*fwd_pos, None)

    def step_forward(self):
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        if fwd_cell is None or fwd_cell.can_overlap():
            self.agent_pos = tuple(fwd_pos)

    def step_lturn(self):
        self.agent_dir = (self.agent_dir - 1 + 4) % 4

    def step_rturn(self):
        self.agent_dir = (self.agent_dir + 1) % 4

    def step_drop(self):
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        if fwd_cell is None and self.carrying:
            self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
            self.carrying.cur_pos = tuple(fwd_pos)
            self.carrying = None

    def step_toggle(self):
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        if fwd_cell:
            fwd_cell.toggle(self, fwd_pos)

    def get_pds_state(self) -> State:
        if self.encoding_executor is None:
            raise RuntimeError(f'Encoding executor is not set. Use set_encoding_executor() to set it.')

        if self.encoding == 'full':
            return _get_pds_state_full(self, self.encoding_executor, ignore_walls=False)
        elif self.encoding == 'basic':
            return _get_pds_state_basic(self, self.encoding_executor, ignore_walls=False)
        else:
            raise ValueError(f'Unknown encoding: {self.encoding}.')

    def debug_print(self):
        print(self)


def _get_pds_state_full(env: MiniGridEnvV20220407, executor: PDSketchExecutor, ignore_walls: bool = False, include_extra_predicates: bool = True):
    """Encode the environment state into a :class:`~concepts.dm.pdsketch.domain.State` object.

    Args:
        env: the environment.
        executor: the executor.
        ignore_walls: whether to ignore walls.
        include_extra_predicates: whether to include extra predicates, including pickable, toggleable, robot-holding.
    """
    domain = executor.domain
    object_names = {'r': domain.types['robot']}
    object_type2id = dict()
    for k in minigrid.OBJECT_TO_IDX:
        object_type2id[k] = 0

    object_images = list()
    object_poses = list()
    objects = list()
    for x, y, obj in env.iter_objects():
        if ignore_walls and obj.type == 'wall':
            continue
        if not hasattr(obj, 'name'):
            obj.name = f'{obj.type}:{object_type2id[obj.type]}'
        object_names[obj.name] = domain.types['item']
        object_images.append(obj.encode())
        object_poses.append((x, y))
        object_type2id[obj.type] += 1
        objects.append(obj)

    state, ctx = executor.new_state(object_names, create_context=True)

    if include_extra_predicates:
        predicates = list()
        for obj in objects:
            if obj.type == 'wall':
                pass
            else:
                predicates.append(ctx.pickable(obj.name))
            if obj.type == 'door':
                predicates.append(ctx.toggleable(obj.name))
        if env.carrying is not None:
            predicates.append(ctx.robot_holding('r', env.carrying.name))
        ctx.define_predicates(predicates)

    ctx.define_feature('robot-pose', torch.tensor([env.agent_pos], dtype=torch.float32))
    ctx.define_feature('robot-direction', torch.tensor([[env.agent_dir]], dtype=torch.int64))
    ctx.define_feature('item-pose', torch.tensor(object_poses, dtype=torch.float32))
    ctx.define_feature('item-image', torch.tensor(object_images, dtype=torch.float32))
    return state


def _get_pds_state_basic(env: MiniGridEnvV20220407, executor: PDSketchExecutor, ignore_walls: bool = False):
    """Encode the environment state into a :class:`~concepts.dm.pdsketch.domain.State` object.

    Args:
        env: the environment.
        executor: the executor.
        ignore_walls: whether to ignore walls.
    """
    domain = executor.domain
    object_names = {'r': domain.types['robot']}
    object_type2id = dict()
    for k in minigrid.OBJECT_TO_IDX:
        object_type2id[k] = 0

    robot_images = list()
    robot_images.append(env.agent_pos + (env.agent_dir, ))

    object_images = list()
    for x, y, obj in env.iter_objects():
        if ignore_walls and obj.type == 'wall':
            continue
        if not hasattr(obj, 'name'):
            obj.name = f'{obj.type}:{object_type2id[obj.type]}'
        object_names[obj.name] = domain.types['item']
        object_images.append(obj.encode() + (x, y))
        object_type2id[obj.type] += 1

    state, ctx = executor.new_state(object_names, create_context=True)
    ctx.define_feature('robot-image', torch.tensor(robot_images, dtype=torch.float32))
    ctx.define_feature('item-image', torch.tensor(object_images, dtype=torch.float32))
    return state


def make_minigrid_env(*args, **kwargs):
    return MiniGridEnvV20220407(*args, **kwargs)


def get_minigrid_domain_filename(encoding: str = 'full') -> str:
    """Get the domain filename of the crafting world."""
    return osp.join(osp.dirname(__file__), 'pds_domains', f'minigrid-domain-v20220407-{encoding}.pdsketch')


def visualize_minigrid_planner(env: MiniGridEnvV20220407, planner):
    torch.set_grad_enabled(False)
    while True:
        init_obs = env.reset()
        state, mission = init_obs['state'], init_obs['mission']
        assert planner is not None
        plan = planner(state, mission)

        cmd = visualize_minigrid_plan(env, plan)
        if cmd == 'q':
            break


def visualize_minigrid_plan(env: MiniGridEnvV20220407, plan):
    env.render()
    print('Plan: ' + ', '.join([str(x) for x in plan]))
    print('Press <Enter> to visualize.')
    _ = input('> ').strip()

    for action in plan:
        print('Executing action: ' + str(action))
        if action.name == 'move':
            pose = action.arguments[1].tensor.tolist()
            dir = action.arguments[2].tensor.item()
            for action in minigrid.find_path(env, pose, dir):
                env.step_inner(action)
                env.render()
                time.sleep(0.5)
        elif action.name == 'forward':
            env.step_inner(MiniGridEnvV20220407.Actions.forward)
        elif action.name == 'lturn':
            env.step_inner(MiniGridEnvV20220407.Actions.left)
        elif action.name == 'rturn':
            env.step_inner(MiniGridEnvV20220407.Actions.right)
        elif action.name == 'pickup':
            env.step_inner(MiniGridEnvV20220407.Actions.pickup)
        elif action.name == 'toggle':
            env.step_inner(MiniGridEnvV20220407.Actions.toggle)
        else:
            raise NotImplementedError(action)
        env.render()
        time.sleep(0.5)

    print('Visualization finished.')
    print('Press <Enter> to continue. Type q to quit.')
    cmd = input('> ').strip()
    return cmd


def _gen_basic_room(env, width, height):
    env.grid = minigrid.Grid(width, height)
    env.agent_pos = (3, 3)
    env.agent_dir = 0
    env.grid.horz_wall(0, 0, 7)
    env.grid.horz_wall(0, 6, 7)
    env.grid.vert_wall(0, 0, 7)
    env.grid.vert_wall(6, 0, 7)


def _gen_grid_goto_single(env, width, height):
    _gen_basic_room(env, width, height)
    objects = list()
    object_poses = list()
    for i in range(1):
        shape = env.np_random.choice([minigrid.Key, minigrid.Box, minigrid.Ball])
        color = env.np_random.choice(minigrid.COLOR_NAMES)

        while True:
            pose = env.np_random.integers(1, 6, size=2)
            if env.grid.get(*pose) is None and not np.all(pose == 3) and not np.all(pose == (4, 3)):  # not initially facing.
                break

        this_object = shape(color)
        objects.append(this_object)
        object_poses.append(pose)
        env.grid.set(*pose, this_object)

    env.goal_obj = goal = env.np_random.choice(objects)
    env.goal_pose = object_poses[objects.index(goal)]
    env.mission = f'(exists (?o - item) (and (robot-is-facing r ?o) (is-{goal.type} ?o)))'

def _gen_grid_goto(env, width, height):
    for _ in range(env.get_option('max_trials', 100)):
        _gen_basic_room(env, width, height)
        objects = list()
        object_poses = list()
        for i in range(env.get_option('nr_objects', 4)):
            shape = env.np_random.choice([minigrid.Key, minigrid.Box, minigrid.Ball])
            color = env.np_random.choice(minigrid.COLOR_NAMES)

            while True:
                pose = env.np_random.integers(1, 6, size=2)
                if env.grid.get(*pose) is None and not np.all(pose == 3) and not np.all(pose == (4, 3)):  # not initially facing.
                    break

            this_object = shape(color)
            objects.append(this_object)
            object_poses.append(pose)
            env.grid.set(*pose, this_object)

        env.goal_obj = goal = env.np_random.choice(objects)
        env.goal_pose = object_poses[objects.index(goal)]
        env.mission = f'(exists (?o - item) (and (robot-is-facing r ?o) (is-{goal.type} ?o) (is-{goal.color} ?o)))'

        path = find_path_to_obj(env, tuple(env.goal_pose))
        if path is not None:
            break

def _gen_grid_pickup(env, width, height):
    for _ in range(env.get_option('max_trials', 100)):
        _gen_basic_room(env, width, height)

        objects = list()
        object_poses = list()
        for i in range(env.get_option('nr_objects', 4)):
            shape = env.np_random.choice([minigrid.Key, minigrid.Box, minigrid.Ball])
            color = env.np_random.choice(minigrid.COLOR_NAMES)

            while True:
                pose = env.np_random.integers(1, 6, size=2)
                if env.grid.get(*pose) is None and not np.all(pose == 3):
                    break

            this_object = shape(color)
            objects.append(this_object)
            object_poses.append(pose)
            env.grid.set(*pose, this_object)
        env.goal_obj = goal = env.np_random.choice(objects)
        env.goal_pose = object_poses[objects.index(goal)]
        env.mission = f'(exists (?o - item) (and (robot-holding r ?o) (is-{goal.type} ?o) (is-{goal.color} ?o)))'

        path = find_path_to_obj(env, tuple(env.goal_pose))
        if path is not None:
            break

def _gen_grid_open(env, width, height):
    _gen_basic_room(env, width, height)

    objects = list()
    for i in range(4):
        color = env.np_random.choice(minigrid.COLOR_NAMES)

        while True:
            pose = env.np_random.integers(1, 6)
            dir = i
            # dir = env.np_random.integers(0, 4)
            if dir == 0:
                pose = (pose, 0)
            elif dir == 1:
                pose = (pose, 6)
            elif dir == 2:
                pose = (0, pose)
            elif dir == 3:
                pose = (6, pose)

            if env.grid.get(*pose).type != 'door':
                break

        this_object = minigrid.Door(color)
        objects.append(this_object)
        env.grid.set(*pose, this_object)

    env.goal_obj = goal = env.np_random.choice(objects)
    env.mission = f'(exists (?o - item) (and (is-open ?o) (is-{goal.color} ?o)))'


def _gen_grid_generalization(env, width, height):
    for _ in range(env.get_option('max_trials', 100)):
        _gen_basic_room(env, width, height)

        objects = list()
        object_poses = list()
        for i in range(env.get_option('nr_objects', 4)):
            shape = env.np_random.choice([minigrid.Key, minigrid.Box, minigrid.Ball])
            color = env.np_random.choice(minigrid.COLOR_NAMES)

            while True:
                pose = env.np_random.integers(1, 6, size=2)
                if env.grid.get(*pose) is None and not np.all(pose == 3):
                    break

            this_object = shape(color)
            objects.append(this_object)
            object_poses.append(pose)
            env.grid.set(*pose, this_object)

        env.goal_obj = goal = env.np_random.choice(objects, size=2, replace=False)
        env.goal_pose = object_poses[objects.index(goal[0])], object_poses[objects.index(goal[1])]
        env.mission = f"""(and
            (exists (?o - item) (and (robot-holding r ?o) (is-{goal[0].type} ?o) (is-{goal[0].color} ?o)))
            (exists (?o - item) (and (robot-is-facing r ?o) (is-{goal[1].type} ?o) (is-{goal[1].color} ?o)))
        )"""

        path = find_path_to_obj(env, tuple(env.goal_pose[0]))
        if path is not None:
            break
        path = find_path_to_obj(env, tuple(env.goal_pose[1]))
        if path is not None:
            break

