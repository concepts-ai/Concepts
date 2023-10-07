#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : constants.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 06/21/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Constants for pymunk.

The following is a table of the constants defined in pymunk.

.. csv-table::
    :header: "Name", "Value", "Description"

    "color_consts.RED", "255, 32, 32, 255", "Red color."
    "color_consts.BLACK", "0, 0, 0, 255", "Black color."
    "color_consts.BLUE", "32, 128, 255, 255", "Blue color."
"""

import argparse

color_consts = argparse.Namespace()
color_consts.RED = (255, 32, 32, 255)
color_consts.BLACK = (0, 0, 0, 255)
color_consts.BLUE = (32, 128, 255, 255)

