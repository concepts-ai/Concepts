#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : typing_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/18/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Utilities for typing.

Table of Custom Types

.. csv-table::
    :header: "Type", "Description"

    ``Tuple2f``, "A tuple of two floats."
    ``Tuple2i``, "A tuple of two integers."
    ``Tuple3f``, "A tuple of three floats."
    ``Tuple3i``, "A tuple of three integers."
    ``Tuple4f``, "A tuple of four floats."
    ``Tuple4i``, "A tuple of four integers."
    ``Tuple5f``, "A tuple of five floats."
    ``Tuple5i``, "A tuple of five integers."
    ``Tuple6f``, "A tuple of six floats."
    ``Tuple6i``, "A tuple of six integers."
    ``Tuple7f``, "A tuple of seven floats."
    ``Tuple7i``, "A tuple of seven integers."
    ``Vec2f``, "A 2D vector of float."
    ``Vec2i``, "A 2D vector of int."
    ``Vec2``, "A 2D vector of float or int."
    ``BoardcastableVec2f``, "A 2D vector of float or a float."
    ``Vec3f``, "A 3D vector of float."
    ``Vec3i``, "A 3D vector of int."
    ``Vec3``, "A 3D vector of float or int."
    ``BoardcastableVec3f``, "A 3D vector of float or a float."
    ``Vec4f``, "A 4D vector of float."
    ``Vec4i``, "A 4D vector of int."
    ``Vec4``, "A 4D vector of float or int."
    ``Vec5f``, "A 5D vector of float."
    ``Vec5i``, "A 5D vector of int."
    ``Vec5``, "A 5D vector of float or int."
    ``Vec6f``, "A 6D vector of float."
    ``Vec6i``, "A 6D vector of int."
    ``Vec6``, "A 6D vector of float or int."
    ``Vec7f``, "A 7D vector of float."
    ``Vec7i``, "A 7D vector of int."
    ``Vec7``, "A 7D vector of float or int."
    ``Color``, "A color, can be a 3D or 4D vector of float or int."
    ``Pos2D``, "A 2D position, can be a 2D vector of float or int."
    ``Pos3D``, "A 3D position, can be a 3D vector of float or int."
    ``Pos6D``, "A 6D position (usually xyz + rpy), can be a 6D vector of float or int."
    ``Pos7D``, "A 7D position (usually xyz + quat), can be a 7D vector of float or int."
"""

import numpy as np
from typing import Union, Tuple, List


Tuple2f = Tuple[float, float]
Tuple2i = Tuple[int, int]
Vec2f = Union[Tuple[float, float], List[float], np.ndarray]
Vec2i = Union[Tuple[int, int], List[int], np.ndarray]
Vec2 = Union[Vec2f, Vec2i]

BraodcastableVec2f = Union[float, Vec2f]

Tuple3f = Tuple[float, float, float]
Tuple3i = Tuple[int, int, int]
Vec3f = Union[Tuple[float, float, float], List[float], np.ndarray]
Vec3i = Union[Tuple[int, int, int], List[int], np.ndarray]
Vec3 = Union[Vec3f, Vec3i]

BroadcastableVec3f = Union[float, Vec3f]

Tuple4f = Tuple[float, float, float, float]
Tuple4i = Tuple[int, int, int, int]
Vec4f = Union[Tuple[float, float, float, float], List[float], np.ndarray]
Vec4i = Union[Tuple[int, int, int, int], List[int], np.ndarray]
Vec4 = Union[Vec4f, Vec4i]

Tuple5f = Tuple[float, float, float, float, float]
Tuple5i = Tuple[int, int, int, int, int]
Vec5f = Union[Tuple[float, float, float, float, float], List[float], np.ndarray]
Vec5i = Union[Tuple[int, int, int, int, int], List[int], np.ndarray]
Vec5 = Union[Vec5f, Vec5i]

Tuple6f = Tuple[float, float, float, float, float, float]
Tuple6i = Tuple[int, int, int, int, int, int]
Vec6f = Union[Tuple[float, float, float, float, float, float], List[float], np.ndarray]
Vec6i = Union[Tuple[int, int, int, int, int, int], List[int], np.ndarray]
Vec6 = Union[Vec6f, Vec6i]

Tuple7f = Tuple[float, float, float, float, float, float, float]
Tuple7i = Tuple[int, int, int, int, int, int, int]
Vec7f = Union[Tuple[float, float, float, float, float, float, float], List[float], np.ndarray]
Vec7i = Union[Tuple[int, int, int, int, int, int, int], List[int], np.ndarray]
Vec7 = Union[Vec7f, Vec7i]

Color = Union[Vec3f, Vec3i, Vec4f, Vec4i]
Pos2D = Union[Vec2f, Vec2i]
Pos3D = Union[Vec3f, Vec3i]
Pos6D = Union[Vec6f, Vec6i]
Pos7D = Union[Vec7f, Vec7i]

