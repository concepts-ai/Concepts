#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : 1-visualize-data.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/5/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import open3d as o3d
import numpy as np
import jacinle
from concepts.vision.george_vision_pipeline.object_centric_vision import CameraTransformation

parser = jacinle.JacArgumentParser()
parser.add_argument('--data', required=True, type='checked_file')
args = parser.parse_args()

def patch_qr():
    import os.path as osp
    import sys

    sys.path.insert(0, osp.expanduser('~/workspace/w-qr/QR/src'))


def main():
    patch_qr()
    data = jacinle.load(args.data)
    jacinle.stprint(data, indent_format='  ')

    while True:
        query = input('Input the index to visualize. Format <camera_index>: ')
        if query == 'exit':
            break
        try:
            camera_index = map(int, query)
        except:
            print('Invalid input.')
            continue

        image_capture = vars(data[0][camera_index])

        rgb = image_capture['rgb_image']
        depth = image_capture['default_depth_image']
        intrinsics, extrinsics = image_capture['camera_intrinsics'], image_capture['camera_extrinsics']

        camera_model = CameraTransformation(intrinsics=intrinsics, extrinsics=extrinsics)
        x, y = np.meshgrid(np.arange(depth.shape[0]), np.arange(depth.shape[1]), indexing='ij')
        x, y, z = camera_model.c2w(
            x, y,
            depth
        )
        xyz = np.stack([x, y, z], axis=-1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3) / 255)

        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, coordinate])


if __name__ == '__main__':
    main()
