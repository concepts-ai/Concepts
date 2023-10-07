#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : collision.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/01/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import pymunk
from typing import Optional, Tuple, List, Set, Dict
from .world import PymunkWorld

__all__ = ['SpacePositionRestorer', 'collision_test']


class SpacePositionRestorer(object):
    def __init__(self, world: PymunkWorld):
        self.world = world

        self.positions = dict()
        for body in self.world.bodies:
            self.positions[body] = body.position

    def restore(self):
        for body, position in self.positions.items():
            body.position = position


def collision_test(world: PymunkWorld, body_positions: Optional[Dict[str, Tuple[float, float]]] = None, bodies: Optional[List[pymunk.Body]] = None) -> Set[Tuple[str, str]]:
    if body_positions is None:
        body_positions = dict()

    restorer = SpacePositionRestorer(world)
    for body, position in body_positions.items():
        body = world.get_body_by_label(body)
        body.position = position

    ret = set()
    shape2body = dict()
    for body in world.bodies:
        for shape in body.shapes:
            shape2body[shape] = body

    if bodies is None:
        bodies = world.bodies
    for body in bodies:
        for shape in body.shapes:
            all_collisions = world.shape_query(shape)
            for other_shape, _ in all_collisions:
                if other_shape is not None and shape2body[other_shape] is not body:
                    ret.add((shape2body[shape].label, shape2body[other_shape].label))

    restorer.restore()
    return ret

