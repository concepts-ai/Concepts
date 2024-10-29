#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : shapely_kinematics.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/28/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""2D shape collision detection using Shapely."""

from typing import Optional, Union, Sequence, Tuple, List, Dict

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, Point


class ShapelyCustomCircle(object):
    def __init__(self, radius: float, center: Tuple[float, float] = (0, 0)):
        self.radius = radius
        self.center = Point(center)


class ShapelyObject(object):
    def __init__(self, label: str, shape: Union[Polygon, ShapelyCustomCircle], center: Point, rotation: float):
        self.label = label
        self.shape = shape
        self.center = center
        self.rotation = rotation
        self.collision_shape = self._compute_collision_shape()

    def _compute_collision_shape(self) -> Union[Polygon, ShapelyCustomCircle]:
        if isinstance(self.shape, ShapelyCustomCircle):
            return ShapelyCustomCircle(radius=self.shape.radius, center=(self.center.x + self.shape.center.x, self.center.y + self.shape.center.y))
        elif isinstance(self.shape, Polygon):
            # Apply the rotation to the shape.
            shape = rotate(self.shape, self.rotation, origin=Point(0, 0), use_radians=True)
            shape = translate(shape, xoff=self.center.x, yoff=self.center.y)
            return shape
        else:
            raise ValueError('Unknown shape type: {}'.format(type(self.shape)))

    def set_pose(self, center: Optional[Tuple[float, float]] = None, rotation: Optional[float] = None):
        if center is not None:
            self.center = Point(center)
        if rotation is not None:
            self.rotation = rotation
        self.collision_shape = self._compute_collision_shape()

    def __str__(self):
        typename = 'Circle' if isinstance(self.shape, ShapelyCustomCircle) else 'Polygon'
        return f'{typename} {self.label} at {self.center} with rotation {self.rotation}'

    def __repr__(self):
        typename = 'Circle' if isinstance(self.shape, ShapelyCustomCircle) else 'Polygon'
        return f'{typename}{{{self.label}, center={self.center}, rotation={self.rotation}}}'


class ShapelyKinematicsSimulator(object):
    def __init__(self):
        self.objects = dict()

    objects: Dict[str, ShapelyObject]
    """The objects in the scene. The key is the name of the object, and the value is the ShapelyObject instance."""

    def add_object(self, label: str, shape: Union[ShapelyCustomCircle, Polygon], center: Optional[Tuple[float, float]] = None, rotation: float = 0):
        if label in self.objects:
            raise NameError(f'The object with label {label} already exists.')

        if center is None:
            center = (0, 0)

        self.objects[label] = ShapelyObject(label=label, shape=shape, center=Point(center), rotation=rotation)

    def add_polygon(self, label: str, vertices: Sequence[Tuple[float, float]], center: Optional[Tuple[float, float]] = None, rotation: float = 0):
        self.add_object(label, Polygon(vertices), center, rotation)

    def add_circle(self, label: str, radius: float, center: Optional[Tuple[float, float]] = None, rotation: float = 0):
        self.add_object(label, ShapelyCustomCircle(radius), center, rotation)

    def get_object_pose(self, label: str) -> Tuple[Point, float]:
        return self.objects[label].center, self.objects[label].rotation

    def set_object_pose(self, label, center=None, rotation=None):
        self.objects[label].set_pose(center, rotation)
        return self.objects[label]

    def pairwise_collision(self, shape_a: Optional[Sequence[ShapelyObject]] = None, shape_b: Optional[Sequence[ShapelyObject]] = None) -> List[Tuple[ShapelyObject, ShapelyObject]]:
        if shape_a is None:
            shape_a = self.objects.values()
        if shape_b is None:
            shape_b = self.objects.values()

        collisions = list()
        for obj_a in shape_a:
            for obj_b in shape_b:
                if obj_a != obj_b:
                    if primitive_collision(obj_a.collision_shape, obj_b.collision_shape):
                        collisions.append((obj_a, obj_b))
        return collisions

    def plot(self, ax: Axes):
        for obj in self.objects.values():
            if isinstance(obj.shape, Polygon):
                ax.plot(*obj.shape.exterior.xy, 'k-')
            elif isinstance(obj.shape, ShapelyCustomCircle):
                ax.add_patch(plt.Circle((obj.center.x, obj.center.y), obj.shape.radius, fill=False, color='k'))


def primitive_collision(shape_a: Union[Polygon, ShapelyCustomCircle], shape_b: Union[Polygon, ShapelyCustomCircle]) -> bool:
    if isinstance(shape_a, Polygon) and isinstance(shape_b, Polygon):
        return shape_a.intersects(shape_b)
    elif isinstance(shape_a, ShapelyCustomCircle) and isinstance(shape_b, ShapelyCustomCircle):
        return Point(shape_a.center).distance(Point(shape_b.center)) < shape_a.radius + shape_b.radius
    elif isinstance(shape_a, Polygon) and isinstance(shape_b, ShapelyCustomCircle):
        return shape_a.distance(Point(shape_b.center)) < shape_b.radius
    elif isinstance(shape_a, ShapelyCustomCircle) and isinstance(shape_b, Polygon):
        return shape_b.distance(Point(shape_a.center)) < shape_a.radius
    else:
        raise ValueError('Unknown shape types: {} and {}'.format(type(shape_a), type(shape_b)))

