#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : 2-visualize-combined-pointcloud.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/5/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import open3d as o3d
import numpy as np
import jacinle

parser = jacinle.JacArgumentParser()
parser.add_argument('--data', required=True, type='checked_file')
args = parser.parse_args()

def patch_qr():
    import os.path as osp
    import sys

    sys.path.insert(0, osp.expanduser('~/workspace/w-qr/QR/src'))


def make_pointcloud_from_rgbd(rgb, depth, intrinsics, extrinsics):
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.flatten()
    y = y.flatten()
    z = depth.flatten()

    points = np.vstack([x, y, np.ones_like(x)])
    points = (np.linalg.inv(intrinsics) @ points) * z
    points = np.vstack([points, np.ones(points.shape[1])])
    points = np.linalg.inv(extrinsics) @ points
    points = points[:3] / points[3]
    return points.T


def main():
    patch_qr()
    data = jacinle.load(args.data)
    jacinle.stprint(data, indent_format='  ')

    print('Found {} cameras.'.format(len(data[0])))

    combined_xyz = list()
    combined_rgb = list()
    for camera_index in range(len(data[0])):
        image_capture = vars(data[0][camera_index])

        rgb = image_capture['rgb_image']
        depth = image_capture['default_depth_image']
        intrinsics, extrinsics = image_capture['camera_intrinsics'], image_capture['camera_extrinsics']

        xyz = make_pointcloud_from_rgbd(rgb, depth, intrinsics, extrinsics)
        # camera_model = CameraTransformation(intrinsics=intrinsics, extrinsics=extrinsics)
        # x, y = np.meshgrid(np.arange(depth.shape[0]), np.arange(depth.shape[1]), indexing='ij')
        # x, y, z = camera_model.c2w(
        #     x, y,
        #     depth
        # )
        # xyz = np.stack([x, y, z], axis=-1)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz)
        # pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3) / 255)

        combined_xyz.append(xyz)
        combined_rgb.append(rgb.reshape(-1, 3) / 255)

    combined_xyz = np.concatenate(combined_xyz, axis=0)
    combined_rgb = np.concatenate(combined_rgb, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_xyz)
    pcd.colors = o3d.utility.Vector3dVector(combined_rgb)

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([pcd, coordinate])


if __name__ == '__main__':
    main()
