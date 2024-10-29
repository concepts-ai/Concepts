#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : namo_polygon_primitives.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/27/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import jacinle
import random
import pymunk
import numpy as np
from queue import PriorityQueue, Queue
from typing import Optional, Sequence, Tuple, List, NamedTuple

from concepts.benchmark.namo.namo_polygon.namo_polygon_env import NamoPolygonEnv, is_point_close_to_body
from concepts.simulator.pymunk.collision import SpacePositionRestorer, collision_test_current


class BoundingBox(NamedTuple):
    x0: int
    y0: int
    x1: int
    y1: int

    def inside(self, x: int, y: int) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def point_inside(self, point: Tuple[int, int]) -> bool:
        return self.inside(point[0], point[1])


def _heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    return float(np.linalg.norm(np.array(pos1) - np.array(pos2), ord=1))


def _cfree_valid(env: NamoPolygonEnv, pos: Tuple[int, int], allow_collision_with_movables: bool = False) -> bool:
    if pos[0] < 0 or pos[0] >= env.world_width or pos[1] < 0 or pos[1] >= env.world_height:
        return False

    env.set_agent_pos(pos)

    if not allow_collision_with_movables:
        return len(collision_test_current(env.world, bodies=[env._agent])) == 0

    collisions = collision_test_current(env.world, bodies=[env._agent])
    for collision in collisions:
        if collision[0].startswith('wall_') or collision[1].startswith('wall_'):
            return False
    return True


def _cfree_valid_with_attached_body(env: NamoPolygonEnv, pos: Tuple[int, int], body: pymunk.Body, pos2: Tuple[int, int]) -> bool:
    env.set_agent_pos(pos)
    body.position = pos2
    rv = len(collision_test_current(env.world, bodies=[env._agent, body])) == 0
    return rv


def find_cfree_path(env: NamoPolygonEnv, goal: Tuple[int, int], goal_atol: float = 2, allow_collision_with_movables: bool = False, bounding_box: Optional[BoundingBox] = None) -> Optional[List[Tuple[int, int]]]:
    agent_pos = env.agent_pos[0], env.agent_pos[1]
    queue = PriorityQueue()
    queue.put((_heuristic(agent_pos, goal), [agent_pos]))
    visited = set()
    visited.add(agent_pos)

    with SpacePositionRestorer(env.world):
        while not queue.empty():
            _, path = queue.get()
            current = path[-1]

            if np.linalg.norm(np.array(current) - np.array(goal), ord=2) < goal_atol:
                return path

            for next_pos in [(current[0] + 1, current[1]), (current[0] - 1, current[1]), (current[0], current[1] + 1), (current[0], current[1] - 1)]:
                if bounding_box is not None and not bounding_box.point_inside(next_pos):
                    continue
                if next_pos not in visited and _cfree_valid(env, next_pos, allow_collision_with_movables=allow_collision_with_movables):
                    new_path = list(path)
                    new_path.append(next_pos)
                    queue.put((_heuristic(next_pos, goal) + len(new_path), new_path))
                    visited.add(next_pos)
        return None


def find_blocking_obstacles(env: NamoPolygonEnv, trajectory: List[Tuple[int, int]]) -> List[pymunk.Body]:
    with SpacePositionRestorer(env.world):
        blocking_obstacles = dict()
        for pos in trajectory:
            env.set_agent_pos(pos)
            collisions = collision_test_current(env.world, bodies=[env._agent])
            for collision in collisions:
                if collision[1].startswith('obstacle_'):
                    if collision[1] not in blocking_obstacles:
                        blocking_obstacles[collision[1]] = env.world.get_body_by_label(collision[1])
        return list(blocking_obstacles.values())


def sample_valid_obstacle_pos(env: NamoPolygonEnv, body: pymunk.Body, nr_samples: int = 100, bounding_box: Optional[BoundingBox] = None, reference_path: Optional[Sequence[Tuple[int, int]]] = None) -> Optional[Tuple[int, int]]:
    with SpacePositionRestorer(env.world):
        for _ in range(nr_samples):
            if bounding_box is not None:
                x = random.randint(bounding_box.x0, bounding_box.x1 - 1)
                y = random.randint(bounding_box.y0, bounding_box.y1 - 1)
            else:
                x = random.randint(0, env.world_width - 1)
                y = random.randint(0, env.world_height - 1)

            body.position = x, y
            env.world.step(1e-6)
            if len(collision_test_current(env.world, bodies=[body])) == 0:
                if reference_path is not None:
                    validated = True
                    for pos in reference_path:
                        env.set_agent_pos(pos)
                        collisions = collision_test_current(env.world, bodies=[env.agent])
                        if body in collisions:
                            validated = False
                    if validated:
                        return x, y
                else:
                    return x, y
    return None


def sample_valid_agent_pos(env: NamoPolygonEnv, nr_samples: int = 100, bounding_box: Optional[BoundingBox] = None) -> Optional[Tuple[int, int]]:
    with SpacePositionRestorer(env.world):
        for _ in range(nr_samples):
            if bounding_box is not None:
                x = random.randint(bounding_box.x0, bounding_box.x1 - 1)
                y = random.randint(bounding_box.y0, bounding_box.y1 - 1)
            else:
                x = random.randint(0, env.world_width - 1)
                y = random.randint(0, env.world_height - 1)

            env.set_agent_pos((x, y))
            env.world.step(1e-6)
            if len(collision_test_current(env.world, bodies=[env.agent])) == 0:
                return x, y
    return None


def find_cfree_path_with_attached_body(env: NamoPolygonEnv, goal: Tuple[int, int], goal_atol: float = 2, bounding_box: Optional[BoundingBox] = None) -> Optional[List[Tuple[int, int]]]:
    body = env.current_attached_body
    body_pos = body.position
    agent_pos = int(env.agent_pos[0]), int(env.agent_pos[1])

    assert is_point_close_to_body(agent_pos, env.current_attached_body)

    queue = PriorityQueue()
    queue.put((_heuristic(body_pos, goal), [agent_pos]))
    visited = set()
    visited.add(body_pos)

    relative_pos = np.array(agent_pos) - np.array(body_pos)

    with SpacePositionRestorer(env.world):
        while not queue.empty():
            _, path = queue.get()
            current = path[-1]
            current_body_pos = np.array(current) - relative_pos

            if np.linalg.norm(np.array(current_body_pos) - np.array(goal), ord=2) < goal_atol:
                return path

            for next_pos in [(current[0] + 1, current[1]), (current[0] - 1, current[1]), (current[0], current[1] + 1), (current[0], current[1] - 1)]:
                if bounding_box is not None and not bounding_box.point_inside(next_pos):
                    continue
                next_body_pos = tuple(np.array(next_pos) - relative_pos)
                if next_body_pos not in visited and _cfree_valid_with_attached_body(env, next_pos, body, next_body_pos):
                    new_path = list(path)
                    new_path.append(next_pos)
                    queue.put((_heuristic(next_body_pos, goal) + len(new_path), new_path))
                    visited.add(next_body_pos)
        return None


def find_reachable_points(env: NamoPolygonEnv, return_path: bool = False, bounding_box: Optional[BoundingBox] = None) -> List[Tuple[int, int]]:
    agent_pos = int(env.agent_pos[0]), int(env.agent_pos[1])
    queue = Queue()
    queue.put(agent_pos)
    outputs = set()
    outputs.add(agent_pos)

    with SpacePositionRestorer(env.world):
        while not queue.empty():
            current = queue.get()

            for next_pos in [(current[0] + 1, current[1]), (current[0] - 1, current[1]), (current[0], current[1] + 1), (current[0], current[1] - 1)]:
                if bounding_box is not None and not bounding_box.point_inside(next_pos):
                    continue
                if next_pos not in outputs and _cfree_valid(env, next_pos):
                    queue.put(next_pos)
                    outputs.add(next_pos)

        return list(outputs)


def find_reachable_points_near(env: NamoPolygonEnv, body: pymunk.Body, max_distance: float = 17, bounding_box: Optional[BoundingBox] = None) -> List[Tuple[int, int]]:
    all_points = find_reachable_points(env, return_path=True, bounding_box=bounding_box)
    all_points = [pos for pos in all_points if is_point_close_to_body(pos, body, max_distance)]
    return all_points


def find_moving_path(env: NamoPolygonEnv, body: pymunk.Body, goal_pos: Tuple[int, int], nr_sample_points: int = 5, goal_atol: float = 2, bounding_box: Optional[BoundingBox] = None, ultimate_goal: Optional[Tuple[int, int]] = None) -> Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
    all_points = find_reachable_points_near(env, body, bounding_box=bounding_box)
    random.shuffle(all_points)

    jacinle.lf_indent_print('Finding an attachment point for the obstacle:', body, 'length:', len(all_points))
    for point1 in all_points[:nr_sample_points]:
        jacinle.lf_indent_print('  Trying to move the obstacle using the attachment point:', point1)
        restorer = SpacePositionRestorer(env.world)
        try:
            with SpacePositionRestorer(env.world):
                path1 = find_cfree_path(env, point1, goal_atol=1, bounding_box=bounding_box)
                if path1 is None:
                    jacinle.lf_indent_print('  No path found for path1')
                    continue

            env.set_agent_pos(path1[-1])
            env.attach_object(body)
            env.world.step(1/1e-6)

            if env.current_attached_body is None:
                continue

            with SpacePositionRestorer(env.world):
                path2 = find_cfree_path_with_attached_body(env, goal_pos, goal_atol=goal_atol, bounding_box=bounding_box)
                if path2 is None:
                    jacinle.lf_indent_print('  No path found for path2')
                    continue

            env.set_agent_pos(path2[-1])
            body.position = goal_pos
            env.world.step(1/1e-6)

            if ultimate_goal is not None:
                path3 = find_cfree_path(env, ultimate_goal, goal_atol=goal_atol, bounding_box=bounding_box)
                if path3 is None:
                    jacinle.lf_indent_print('  No path found for path3')
                    continue
                return path1, path2
            else:
                if path2 is not None:
                    return path1, path2
        finally:
            restorer.restore()
            env.detach_object()

