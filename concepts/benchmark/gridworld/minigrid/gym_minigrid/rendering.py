#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : rendering.py
# Author : Zhezheng Luo, Jiayuan Mao
# Email  : ezzluo@mit.edu, jiayuanm@mit.edu
# Date   : 04/23/2021
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

r"""
Original file from Chevalier-Boisvert, Maxime and Willems, Lucas and Pal, Suman.

.. code-block:: bibtex

    @misc{gym_minigrid,
      author = {Chevalier-Boisvert, Maxime and Willems, Lucas and Pal, Suman},
      title = {Minimalistic Gridworld Environment for OpenAI Gym},
      year = {2018},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/maximecb/gym-minigrid}},
    }
"""

import math
import numpy as np
from typing import Optional, Callable, Tuple

Vector2f = Tuple[float, float]
Vector3f = Tuple[float, float, float]

__all__ = ['downsample', 'fill_coords', 'rotate_fn', 'point_in_line', 'point_in_circle', 'point_in_rect', 'point_in_triangle', 'highlight_img']


def downsample(img: np.ndarray, factor: int) -> np.ndarray:
    """Downsample an image along both dimensions by some factor."""
    assert img.shape[0] % factor == 0
    assert img.shape[1] % factor == 0

    img = img.reshape([img.shape[0] // factor, factor, img.shape[1] // factor, factor, 3])
    img = img.mean(axis=3)
    img = img.mean(axis=1)

    return img


def fill_coords(img: np.ndarray, fn: Callable[[float, float], bool], color: Tuple[int, int, int]):
    """Fill pixels of an image with coordinates matching a filter function."""
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                img[y, x] = color
    return img


def rotate_fn(fin: Callable[[float, float], bool], cx: float, cy: float, theta: float) -> Callable[[float, float], bool]:
    def fout(x, y):
        x = x - cx
        y = y - cy
        x2 = cx + x * math.cos(-theta) - y * math.sin(-theta)
        y2 = cy + y * math.cos(-theta) + x * math.sin(-theta)
        return fin(x2, y2)
    return fout


def point_in_line(x0: float, y0: float, x1: float, y1: float, r: float) -> Callable[[float, float], bool]:
    p0 = np.array([x0, y0])
    p1 = np.array([x1, y1])
    dir = p1 - p0
    dist = np.linalg.norm(dir)
    dir = dir / dist

    xmin = min(x0, x1) - r
    xmax = max(x0, x1) + r
    ymin = min(y0, y1) - r
    ymax = max(y0, y1) + r

    def fn(x, y):
        # Fast, early escape test
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False

        q = np.array([x, y])
        pq = q - p0

        # Closest point on line
        a = np.dot(pq, dir)
        a = np.clip(a, 0, dist)
        p = p0 + a * dir

        dist_to_line = np.linalg.norm(q - p)
        return dist_to_line <= r

    return fn


def point_in_circle(cx: float, cy: float, r: float) -> Callable[[float, float], bool]:
    def fn(x, y):
        return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r
    return fn


def point_in_rect(xmin: float, xmax: float, ymin: float, ymax: float) -> Callable[[float, float], bool]:
    def fn(x, y):
        return xmin <= x <= xmax and ymin <= y <= ymax
    return fn


def point_in_triangle(a: Vector2f, b: Vector2f, c: Vector2f) -> Callable[[float, float], bool]:
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    def fn(x, y):
        v0 = c - a
        v1 = b - a
        v2 = np.array((x, y)) - a

        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u + v) < 1

    return fn


def highlight_img(img_: np.ndarray, color: Optional[Vector3f] = (255, 255, 255), alpha: Optional[float] = 0.30):
    """ Add highlighting to an image."""
    blend_img = img_ + alpha * (np.array(color, dtype=np.uint8) - img_)
    blend_img = blend_img.clip(0, 255).astype(np.uint8)
    img_[:, :, :] = blend_img
