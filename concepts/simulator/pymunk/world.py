#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : world.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 06/21/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import pymunk
import random
from typing import Any, Optional, Tuple, Dict

from concepts.algorithm.configuration_space import BoxConfigurationSpace, CollisionFreeProblemSpace
from concepts.utils.range import Range

__all__ = ['PymunkWorld', 'PymunkSingleObjectConfigurationSpace', 'PymunkCollisionFreeProblemSpace']


class PymunkWorld(pymunk.Space):
    """A wrapper :class:`pymunk.Space` providing a manager for screen size and body labels."""

    def __init__(self, *args, screen_width: Optional[int] = None, screen_height: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.body2label = dict()
        self.label2body = dict()
        self.body_selectable = dict()
        self.selectable_bodies = list()

    def add_body(self, body, selectable=False, label=None) -> pymunk.Body:
        """Add a body to the space.

        Args:
            body: the body to be added.
            selectable: whether the body is selectable.
            label: the label of the body.

        Returns:
            The body added.
        """
        self.add(body, *body.shapes)
        self.label2body[label] = body
        self.body2label[body] = label
        self.body_selectable[body] = selectable

        body.label = label
        body.selectable = selectable
        if selectable:
            self.selectable_bodies.append(body)

        return body

    @property
    def bodies_extra(self):
        for body in self.bodies:
            yield body, self.body_selectable[body], self.body2label[body]

    def select_body(self, point: Tuple[float, float]) -> Optional[pymunk.Body]:
        import concepts.simulator.pymunk.body_utils as body_utils
        return body_utils.select_body(self, point, self.selectable_bodies)

    def get_body_by_label(self, name: str) -> pymunk.Body:
        assert name in self.label2body, f'Body "{name}" is not in the space.'
        return self.label2body[name]

    def random_body_pos(self, body: Optional[pymunk.Body] = None) -> Tuple[float, float]:
        import concepts.simulator.pymunk.body_utils as body_utils
        if body is None:
            return random.randint(0, self.screen_width), random.randint(0, self.screen_height)
        else:
            return body_utils.random_body_pos(self, body)

    def get_body_poses(self) -> Dict[str, Tuple[float, float]]:
        return {label: tuple(body.position) for label, body in self.label2body.items()}

    def get_body_states(self) -> Dict[str, Dict[str, Any]]:
        return {label: {
            'position': tuple(body.position),
            'velocity': tuple(body.velocity),
            'angle': body.angle,
            'angular_velocity': body.angular_velocity
        } for label, body in self.label2body.items()}

    def get_collision_free_pspace(self, controlling_object: str, ignore_collision_filter=None, max_diff=2) -> 'PymunkCollisionFreeProblemSpace':
        return PymunkCollisionFreeProblemSpace(
            PymunkSingleObjectConfigurationSpace(self, controlling_object, max_diff=max_diff),
            ignore_collision_filter=ignore_collision_filter
        )


class PymunkSingleObjectConfigurationSpace(BoxConfigurationSpace):
    def __init__(self, space: PymunkWorld, controlling_object: str, max_diff: float = 2):
        super().__init__([Range(0, space.screen_width), Range(space.screen_height // 2 - 200, space.screen_height)], max_diff)
        self.controlling_object = controlling_object
        self.pymunk_space = space


class PymunkCollisionFreeProblemSpace(CollisionFreeProblemSpace):
    def __init__(self, cspace: PymunkSingleObjectConfigurationSpace, ignore_collision_filter=None):
        super().__init__(cspace)
        self.ignore_collision_filter = ignore_collision_filter

    @property
    def space(self) -> PymunkWorld:
        return self.cspace.pymunk_space

    @property
    def controlling_object(self) -> str:
        return self.cspace.controlling_object

    def collide(self, configuration):
        from .collision import collision_test
        all_collisions = collision_test(self.space, {self.cspace.controlling_object: configuration}, bodies=[self.space.get_body_by_label(self.cspace.controlling_object)])
        if self.ignore_collision_filter is not None:
            all_collisions = [c for c in all_collisions if not self.ignore_collision_filter(*c)]
        return len(all_collisions) > 0

