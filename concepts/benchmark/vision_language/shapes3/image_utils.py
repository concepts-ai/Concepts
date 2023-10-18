#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : image_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/18/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
import numpy.random as npr
import cv2

g_shapes_index_to_name = {0: 'circle', 1: 'triangle', 2: 'rectangle'}
g_colors_index_to_name = {0: 'red', 1: 'green', 2: 'blue'}


def create_shapes3(object_size: int = 32):
    canvas_size = (object_size, object_size * 3)  # h x w
    canvas = np.zeros(canvas_size + (3, ), dtype=np.uint8)
    shapes = list()

    for i in range(3):
        shape = npr.randint(0, 3)  # 0: circle, 1: triangle, 2: rectangle
        color = npr.randint(0, 3)  # 0: red, 1: green, 2: blue

        shapes.append({'shape': g_shapes_index_to_name[shape], 'color': g_colors_index_to_name[color]})

        if color == 0:
            color = (0, 0, 200)
        elif color == 1:
            color = (0, 200, 0)
        else:
            color = (200, 0, 0)

        if shape == 0:
            radius = int(object_size * 0.4)
            center = (object_size // 2 + i * object_size, object_size // 2)
            canvas = cv2.circle(canvas, center, radius, color, -1)
        elif shape == 1:
            pts = np.array([
                [object_size // 2 + i * object_size, object_size // 4],
                [object_size // 4 + i * object_size, object_size * 3 // 4],
                [object_size * 3 // 4 + i * object_size, object_size * 3 // 4]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            canvas = cv2.fillPoly(canvas, [pts], color)
        else:
            pts = np.array([
                [object_size // 4 + i * object_size, object_size // 4],
                [object_size // 4 + i * object_size, object_size * 3 // 4],
                [object_size * 3 // 4 + i * object_size, object_size * 3 // 4],
                [object_size * 3 // 4 + i * object_size, object_size // 4]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            canvas = cv2.fillPoly(canvas, [pts], color)

    return canvas, shapes


