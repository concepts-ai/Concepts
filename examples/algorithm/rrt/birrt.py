#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : birrt.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/28/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
import matplotlib.pyplot as plt
from concepts.algorithm.configuration_space import BoxConfigurationSpace, CollisionFreeProblemSpace
from concepts.algorithm.rrt.rrt import birrt
from concepts.math.range import Range
from concepts.simulator.shapely_kinematics.shapely_kinematics import ShapelyKinematicsSimulator, ShapelyCustomCircle, Polygon


def make_shapely_world():
    world = ShapelyKinematicsSimulator()
    world.add_circle('robot', radius=0.5, center=(-3, -3))
    world.add_polygon('obstacle1', vertices=[(-1, -1), (1, -1), (1, 1), (-1, 1)])
    return world


def main():
    world = make_shapely_world()
    space = BoxConfigurationSpace([Range(-5, 5), Range(-5, 5)], 0.1)

    def is_colliding(q):
        pos = world.get_object_pose('robot')[0]
        world.set_object_pose('robot', center=q, rotation=0)
        rv = world.pairwise_collision()
        world.set_object_pose('robot', center=pos, rotation=0)

        return len(rv) > 0

    # Define the problem space.
    problem_space = CollisionFreeProblemSpace(space, is_colliding)

    # Define the start and goal configurations.
    start = (-3, -3)
    goal = (3, 3)

    # Solve the problem.
    path, _ = birrt(problem_space, start, goal, smooth_fine_path=True)
    print(path)

    # Visualize the result.
    fig, ax = plt.subplots(figsize=(6, 6))
    world.plot(ax)
    ax.plot(*zip(*path), 'r-')
    ax.plot(*start, 'go')
    ax.plot(*goal, 'bo')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    plt.show()


if __name__ == '__main__':
    main()

