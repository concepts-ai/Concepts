#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : namo_polygon_env.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/26/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import pygame
from pymunk import Body
from typing import Optional, Any, Sequence, Tuple, NamedTuple, Dict

import numpy as np
import jacinle.io as io

import concepts.simulator.pymunk.body_utils as body_utils
from concepts.simulator.pymunk.default_env import PymunkDefaultEnv
from concepts.simulator.pymunk.collision import SpacePositionRestorer
from concepts.simulator.pymunk.collision import collision_test_current


class NamoPolygonExecutionResult(NamedTuple):
    success: bool
    episode_end: bool
    episode_success: bool


class NamoPolygonEnvState(NamedTuple):
    simulation_state: Dict[str, Any]
    current_attached_body: Optional[Body]
    current_attached_body_relative_pos: Optional[Tuple[float, float]]

    @classmethod
    def make(cls, env: 'NamoPolygonEnv') -> 'NamoPolygonEnvState':
        saved = SpacePositionRestorer(env.world).positions
        return cls(
            simulation_state={k.label: v for k, v in saved.items()},
            current_attached_body=env.current_attached_body,
            current_attached_body_relative_pos=env.current_attached_body_relative_pos
        )

    def restore(self, env: 'NamoPolygonEnv'):
        for k, v in self.simulation_state.items():
            env.world.get_body_by_label(k).position = v
        env.set_current_attached_body(self.current_attached_body)
        env.current_attached_body_relative_pos = self.current_attached_body_relative_pos
        env.world.step(1e-6)


class NamoPolygonEnv(PymunkDefaultEnv):
    def __init__(self, scene_json, **kwargs):
        self.scene_json = scene_json
        self.scene = io.load_json(scene_json)

        kwargs.setdefault('damping', 0.0)
        kwargs.setdefault('world_width', self.scene['width'])
        kwargs.setdefault('world_height', self.scene['height'])
        kwargs.setdefault('action_velocity', 1000)
        super().__init__(**kwargs)

        self._agent = None
        self._obstacles = list()
        self.current_attached_body = None
        self.current_attached_body_relative_pos = None

        self.set_additional_step_callable(self._handle_attached_body)

    def reset_scene(self):
        if self.scene['start'] is not None:
            self._agent = body_utils.add_ball(self.world, label='agent', pos=self.scene['start'], radius=8, selectable=True)
            self.set_current_selection(self._agent)
        if self.scene['goal'] is not None:
            self.add_additional_drawing_region_circle(self.scene['goal'], 15, (0, 255, 0))

        # Walls
        for i, obs in enumerate(self.scene['raw_walls']):
            center, vertices = center_polygon(obs)
            body_utils.add_polygon(self.world, vertices, pos=center, movable=False, label=f'wall_{i}', use_convex_decomposition=True, color=(0, 0, 0, 255))

        # Obstacles
        self._obstacles = list()
        for i, obs in enumerate(self.scene['obstacles']):
            center, vertices = center_polygon(obs)
            obstacle = body_utils.add_polygon(self.world, vertices, pos=center, movable=True, color=(128, 128, 255, 255), label=f'obstacle_{i}', use_convex_decomposition=True)
            self._obstacles.append(obstacle)

        self.set_current_attached_body(None)

    def set_current_attached_body(self, body: Optional[Body]):
        self.current_attached_body = body
        if body is not None:
            self.current_attached_body_relative_pos = np.array(self.agent_pos) - np.array(body.position)

    def _handle_attached_body(self, env):
        if self.current_attached_body is not None:
            self.current_attached_body.position = tuple(np.array(self.agent_pos) - self.current_attached_body_relative_pos)

    @property
    def agent(self):
        return self._agent

    @property
    def agent_start(self) -> Tuple[float, float]:
        return self.scene['start']

    @property
    def agent_goal(self) -> Tuple[float, float]:
        return self.scene['goal']

    @property
    def agent_pos(self):
        return self._agent.position

    def set_agent_pos(self, pos: Tuple[float, float]):
        self._agent.position = pos

        if self.current_attached_body is not None:
            self.current_attached_body.position = tuple(np.array(pos) - self.current_attached_body_relative_pos)

    @property
    def obstacles(self) -> Sequence[Body]:
        return self._obstacles

    def move_cfree_trajectory(self, trajectory, render_mode: str = 'human') -> NamoPolygonExecutionResult:
        for pos in trajectory:
            if np.linalg.norm(np.array(pos) - self._agent.position) < 2:
                self.set_agent_pos(pos)
                self.step(1, render_mode=render_mode)
            if collision_test_current(self.world, [self._agent]):
                return NamoPolygonExecutionResult(success=False, episode_end=True, episode_success=False)

        episode_end, episode_success = self._check_episode_end()
        return NamoPolygonExecutionResult(success=True, episode_end=episode_end, episode_success=episode_success)

    def attach_object(self, body: Body, render_mode: str = 'human') -> NamoPolygonExecutionResult:
        if is_point_close_to_body(self.agent_pos, body):
            self.set_current_attached_body(body)
            self.step(1, render_mode=render_mode)
            return NamoPolygonExecutionResult(success=True, episode_end=False, episode_success=False)
        return NamoPolygonExecutionResult(success=False, episode_end=False, episode_success=False)

    def move_cfree_trajectory_with_attached_object(self, trajectory, render_mode: str = 'human') -> NamoPolygonExecutionResult:
        if self.current_attached_body is None:
            return NamoPolygonExecutionResult(success=False, episode_end=True, episode_success=False)

        relative_pos = np.array(self.agent_pos) - np.array(self.current_attached_body.position)
        for pos in trajectory:
            if np.linalg.norm(np.array(pos) - self._agent.position) < 2:
                self.set_agent_pos(pos)
                self.current_attached_body.position = tuple(np.array(pos) - relative_pos)
                self.step(1, render_mode=render_mode)
            if collision_test_current(self.world, [self._agent, self.current_attached_body]):
                return NamoPolygonExecutionResult(success=False, episode_end=True, episode_success=False)

        episode_end, episode_success = self._check_episode_end()
        return NamoPolygonExecutionResult(success=True, episode_end=episode_end, episode_success=episode_success)

    def detach_object(self, render_mode: str = 'human') -> NamoPolygonExecutionResult:
        self.set_current_attached_body(None)
        self.step(1, render_mode=render_mode)
        return NamoPolygonExecutionResult(success=True, episode_end=False, episode_success=False)

    def _check_episode_end(self) -> Tuple[bool, bool]:
        return False, np.linalg.norm(np.array(self.agent_pos) - np.array(self.scene['goal'])) < 10

    def distance_to_body(self, body: Body) -> float:
        distances = list()
        for shape in body.shapes:
            distances.append(shape.point_query(self.agent_pos).distance)
        return min(distances)

    def _handle_keyboard_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                for obs in self._obstacles:
                    if is_point_close_to_body(self.agent_pos, obs):
                        self.set_current_attached_body(obs)
                        print('Attached to obstacle:', obs.label, obs)
                        break
            elif event.key == pygame.K_d:
                self.set_current_attached_body(None)
                print('Detached from obstacles.')
            else:
                print('New obstacle pos:', self.current_attached_body.position if self.current_attached_body is not None else None)


def center_polygon(vertices) -> Tuple[Tuple[float, float], Sequence[Tuple[float, float]]]:
    vertices = np.array(vertices)
    center = np.mean(vertices, axis=0)
    return center.tolist(), (vertices - center).tolist()


def is_point_close_to_body(point: Tuple[int, int], body: Body, max_distance: float = 20) -> bool:
    for shape in body.shapes:
        if shape.point_query(point).distance < max_distance:
            return True
    return False

