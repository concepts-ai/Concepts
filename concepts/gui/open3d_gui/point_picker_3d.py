#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : point_picker_3d.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/13/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import open3d as o3d

__all__ = ['Open3DPointPicker', 'open3d_point_picker']


class Open3DPointPicker(object):
    def __init__(self):
        self.vis = o3d.visualization.VisualizerWithEditing()
        self.vis.create_window()
        self.registered_points = list()

    def run(self, geometry: o3d.geometry.PointCloud):
        self.print_help()
        self.vis.add_geometry(geometry)
        self.vis.run()
        self.vis.destroy_window()
        self.registered_points = self.vis.get_picked_points()
        return self.registered_points

    def print_help(self):
        print("")
        print('-' * 80)
        print("1) In order to pick a point, use [shift + left click]")
        print("2) To undo point picking, use [shift + right click]")
        print("3) After picking points, press 'q' to close the window")
        print('-' * 80)
        print("")


def open3d_point_picker(pcd: o3d.geometry.PointCloud, **kwargs):
    picker = Open3DPointPicker()
    return picker.run(pcd, **kwargs)
