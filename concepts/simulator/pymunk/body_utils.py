#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : body_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 06/21/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import random

import numpy as np
import pymunk
from typing import Optional, Iterable, Tuple, Sequence

from concepts.utils.typing_utils import Vec4i
from .world import PymunkWorld
from .constants import color_consts


__all__ = [
    'get_screen_size', 'get_body_bbox',
    'add_ball', 'add_box',
    'add_shape_I', 'add_shape_L', 'add_shape_T', 'add_shape_C',
    'random_body_pos',
    'select_body'
]


def get_screen_size() -> Tuple[int, int]:
    import pygame
    screen_width, screen_height = pygame.display.get_window_size()
    return screen_width, screen_height


def get_body_bbox(body: pymunk.Body) -> pymunk.BB:
    bb = pymunk.BB()
    for s in body.shapes:
        bb = bb.merge(s.cache_bb())
    return bb


def convex_decomposition(vertices: Sequence[Tuple[float, float]]) -> Sequence[Sequence[Tuple[float, float]]]:
    from py2d.Math.Polygon import Polygon

    poly = Polygon.from_tuples(vertices)
    parts = Polygon.convex_decompose(poly)
    return [part.as_tuple_list() for part in parts]


def add_polygon(
    world: PymunkWorld, vertices: Sequence[Tuple[float, float]],
    mass: float = 1.0, pos: Optional[Tuple[float, float]] = None,
    movable: bool = True, color: Vec4i = color_consts.BLACK,
    use_convex_decomposition: bool = False, **kwargs
):
    if use_convex_decomposition:
        inertia = pymunk.moment_for_poly(mass, vertices)
        body = pymunk.Body(mass, inertia, body_type=pymunk.Body.DYNAMIC if movable else pymunk.Body.STATIC)
        shapes = list()
        for part in convex_decomposition(vertices):
            shape = pymunk.Poly(body, part)
            shape.color = color
            shape.friction = 1.0
            shape.elasticity = 0.8
            shapes.append(shape)
        body.position = pos if pos is not None else random_body_pos(world, body)
        return world.add_body(body, shapes=shapes, **kwargs)
    else:
        inertia = pymunk.moment_for_poly(mass, vertices)
        body = pymunk.Body(mass, inertia, body_type=pymunk.Body.DYNAMIC if movable else pymunk.Body.STATIC)
        shape = pymunk.Poly(body, vertices)
        shape.color = color
        shape.friction = 1.0
        shape.elasticity = 0.8

        body.position = pos if pos is not None else random_body_pos(world, body)
        return world.add_body(body, **kwargs)


def add_ball(world: PymunkWorld, mass: float = 1.0, radius: float = 14.0, pos: Optional[Tuple[float, float]] = None, kinematic=False, **kwargs):
    inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
    body = pymunk.Body(mass, inertia, body_type=pymunk.Body.KINEMATIC if kinematic else pymunk.Body.DYNAMIC)
    shape = pymunk.Circle(body, radius, (0, 0))
    shape.color = color_consts.RED
    shape.friction = 2.0
    shape.elasticity = 0

    body.position = pos if pos is not None else random_body_pos(world, body)
    return world.add_body(body, **kwargs)


def add_box(world: PymunkWorld, mass: float = 1.0, half_length: float = 14.0, pos: Optional[Tuple[float, float]] = None, static=False, friction=2.0, **kwargs):
    inertia = pymunk.moment_for_box(mass, (half_length * 2, half_length * 2))
    body = pymunk.Body(mass, inertia, body_type=pymunk.Body.STATIC if static else pymunk.Body.DYNAMIC)
    shape = pymunk.Poly(body, [(-half_length, -half_length), (half_length, -half_length), (half_length, half_length), (-half_length, half_length)])
    shape.color = color_consts.BLUE
    shape.friction = friction
    shape.elasticity = 0

    body.position = pos if pos is not None else random_body_pos(world, body)
    return world.add_body(body, **kwargs)


def add_ground(world: PymunkWorld, mass: float = 1.0, half_length: float = 14.0, pos: Optional[Tuple[float, float]] = None, static=False, **kwargs):
    inertia = pymunk.moment_for_box(mass, (half_length * 2, half_length * 2))
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    shape = pymunk.Poly(body, [(-half_length, -half_length), (half_length, -half_length), (half_length, half_length), (-half_length, half_length)])
    shape.color = color_consts.GRAY
    shape.friction = 1.0
    shape.elasticity = 0.5

    body.position = pos if pos is not None else random_body_pos(world, body)
    return world.add_body(body, **kwargs)


def add_rectangle(world: PymunkWorld, mass: float = 1.0, width: float = 14.0, height: float = 14.0, radius: float = 2.0, pos: Optional[Tuple[float, float]] = None, orientation: float = 0.0, **kwargs):
    """Add a rectangle to the world.

    Args:
        world: the world to add the rectangle to.
        mass: the mass of the rectangle.
        width: the width of the rectangle (along the x-axis).
        height: the height of the rectangle (along the y-axis).
        radius: the radius of the rectangle's corners.
        pos: the position of the rectangle.
        orientation: the orientation of the rectangle, in degrees.
        **kwargs: additional arguments to pass to the world.add_body method.
    """

    inertia = pymunk.moment_for_box(mass, (width, height))
    body = pymunk.Body(mass, inertia)
    shape = pymunk.Poly.create_box(body, (width, height), radius=radius)
    shape.color = color_consts.BLUE
    shape.friction = 1.0
    shape.elasticity = 0.8

    body.position = pos if pos is not None else random_body_pos(world, body)
    body.angle = np.deg2rad(orientation)
    return world.add_body(body, **kwargs)


def add_shape_I(world: PymunkWorld, length: float = 200, thickness: float = 3, pos: Optional[Tuple[float, float]] = None, **kwargs):
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    shape = pymunk.Segment(body, (0, 0), (0, length), thickness)
    shape.color = color_consts.BLACK
    shape.friction = 1.0

    body.position = pos if pos is not None else random_body_pos(world, body)
    return world.add_body(body, **kwargs)


def add_shape_L(world: PymunkWorld, length1: float = 200, length2: float = 50, thickness: float = 3, pos: Optional[Tuple[float, float]] = None, **kwargs):
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    shape1 = pymunk.Segment(body, (0, 0), (0, length1), thickness)
    shape1.color = color_consts.BLACK
    shape1.friction = 1.0
    shape2 = pymunk.Segment(body, (0, 0), (length2, 0), thickness)
    shape2.color = color_consts.BLACK
    shape2.friction = 1.0

    body.position = pos if pos is not None else random_body_pos(world, body)
    return world.add_body(body, **kwargs)


def add_shape_T(world: PymunkWorld, length1: float = 200, length2: float = 50, thickness: float = 3, pos: Optional[Tuple[float, float]] = None, dynamic=False, **kwargs):
    if dynamic:
        body = pymunk.Body(body_type=pymunk.Body.DYNAMIC, mass=1.0, moment=100)
    else:
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)

    shape1 = pymunk.Segment(body, (0, 0), (0, length1), thickness)
    shape1.color = color_consts.BLACK
    shape1.friction = 1.0
    shape2 = pymunk.Segment(body, (-length2, 0), (+length2, 0), thickness)
    shape2.color = color_consts.BLACK
    shape2.friction = 1.0

    body.position = pos if pos is not None else random_body_pos(world, body)
    return world.add_body(body, **kwargs)


def add_shape_C(world: PymunkWorld, length1: float = 200, length2: float = 50, length3: float = 25, thickness: float = 3, pos: Optional[Tuple[float, float]] = None, **kwargs):
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    shape1 = pymunk.Segment(body, (0, 0), (0, length1), thickness)
    shape1.color = color_consts.BLACK
    shape2 = pymunk.Segment(body, (-length2, 0), (+length2, 0), thickness)
    shape2.color = color_consts.BLACK
    shape3 = pymunk.Segment(body, (-length2, 0), (-length2, -length3), thickness)
    shape3.color = color_consts.BLACK
    shape4 = pymunk.Segment(body, (+length2, 0), (+length2, -length3), thickness)
    shape4.color = color_consts.BLACK

    body.position = pos if pos is not None else random_body_pos(world, body)
    return world.add_body(body, **kwargs)


def random_body_pos(world: PymunkWorld, body: pymunk.Body) -> Tuple[float, float]:
    bb = get_body_bbox(body)
    obj_width = int(bb.right - bb.left) // 2 + 1
    obj_height = int(bb.top - bb.bottom) // 2 + 1
    return (random.randint(obj_width, world.screen_width - obj_width), random.randint(obj_height, world.screen_height - obj_height))


def select_body(world: PymunkWorld, pos: Tuple[float, float], selectable_bodies: Optional[Iterable[pymunk.Body]] = None):
    if selectable_bodies is None:
        selectable_bodies = world.bodies

    for b in selectable_bodies:
        for s in b.shapes:
            info = s.point_query(pos)
            if info.distance < 0:
                return b
    return None

