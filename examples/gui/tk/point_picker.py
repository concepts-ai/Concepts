#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : point_picker.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/29/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp
from typing import Tuple
from concepts.gui.tk.point_picker import get_click_coordinates_from_image_path


def main():
    image_path = osp.join(
        osp.dirname(osp.dirname(__file__)),
        'infinite-corridor.jpg'
    )

    coords = get_click_coordinates_from_image_path(image_path, min_dimension=800)
    print('Clicked coordinates:', coords)


if __name__ == '__main__':
    main()
