#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : 3-visualize-smoothed-combined-pointcloud.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/05/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import torch
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import jacinle

from typing import Optional, Tuple
from concepts.vision.depth_smoother.depth_anything_smoother import DepthAnythingV2Smoother, DepthAnythingV2SmootherConfig


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


def smooth(
    smoother: DepthAnythingV2Smoother, rgb: np.ndarray, depth: np.ndarray,
    mask: Optional[np.ndarray] = None, *,
    depth_range_filter: Optional[Tuple[float, float]] = None,
    visualize_fitting: bool = False, visualize_predictions: bool = False, visualize_distributions: bool = False, visualization_title=None
):
    if depth_range_filter is not None:
        new_mask = (depth > depth_range_filter[0]) & (depth < depth_range_filter[1])
        if mask is not None:
            new_mask &= mask
        mask = new_mask

    new_depth = smoother(rgb, depth, mask, visualize=visualize_fitting)

    # Use the new_depth to filter the depth.

    error = depth - new_depth
    # Run a 3x3 median filter to figure out the "expected" error at each pixel.

    import cv2
    expected_error = cv2.medianBlur(error.astype('float32'), 3)

    # If the error is too different from the expected error, we do not trust the depth from the point.

    mask = np.abs(error - expected_error) < 0.01

    print(f'Filtered out {np.sum(~mask)} points out of {np.prod(depth.shape)}.')

    rv_depth = new_depth.copy()
    rv_depth[~mask] = 0.0

    if visualize_predictions:
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        axes[0].imshow(rgb)
        axes[1].imshow(depth)
        axes[2].imshow(new_depth)
        if visualization_title is not None:
            plt.suptitle(visualization_title)
        plt.show()

    # Plot the distribution of the raw depth vs. the smoothed depth
    if visualize_distributions:
        plt.figure()
        indices = np.random.choice(np.prod(depth.shape), 1000)
        plt.scatter(depth.flatten()[indices], new_depth.flatten()[indices], s=1)
        plt.plot([0, 10], [0, 10], 'r--')
        plt.xlabel('Raw Depth')
        plt.ylabel('Smoothed Depth')
        plt.title('Depth Distribution for ' + visualization_title)
        plt.show()

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(10, 5))
    axes[0].imshow(rgb)
    axes[1].imshow(depth)
    axes[2].imshow(new_depth)
    axes[3].imshow(rv_depth)
    plt.show()

    return rv_depth, mask


def main():
    patch_qr()
    data = jacinle.load(args.data)
    jacinle.stprint(data, indent_format='  ')
    smoother = DepthAnythingV2Smoother(DepthAnythingV2SmootherConfig(curve_smooth=10, model_size='large'))

    print('Found {} cameras.'.format(len(data[0])))

    combined_xyz = list()
    combined_rgb = list()
    new_combined_xyz = list()
    new_combined_rgb = list()
    for camera_index in range(len(data[0])):
        image_capture = vars(data[0][camera_index])

        rgb = image_capture['rgb_image']
        depth = image_capture['default_depth_image']
        intrinsics, extrinsics = image_capture['camera_intrinsics'], image_capture['camera_extrinsics']

        new_depth, inferred_mask = smooth(smoother, rgb, depth, depth_range_filter=(0.2, 1.2), visualize_fitting=False, visualization_title=f'Camera {camera_index}')

        xyz = make_pointcloud_from_rgbd(rgb, depth, intrinsics, extrinsics)
        # depth_flatten = depth.flatten()
        # mask = (0.2 < depth_flatten) & (depth_flatten < 1.2)
        # mask_rgb = np.zeros((xyz.shape[0], 3)) + [0, 0, 0.5]
        # mask_rgb[~inferred_mask.flatten(), :] = [0.5, 0, 0]
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz[mask])
        # pcd.colors = o3d.utility.Vector3dVector(mask_rgb[mask])
        # print('Visualizing what have been filtered out.')
        # o3d.visualization.draw_geometries([pcd])
        print(f'Camera {camera_index}: {xyz.shape[0]} points.')
        depth_flatten = depth.flatten()
        mask = (0.2 < depth_flatten) & (depth_flatten < 1.2)
        xyz = xyz[mask]
        rgb_flatten = rgb.reshape(-1, 3)[mask] / 255

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb_flatten)

        # Remove outliners
        pcd, indices = pcd.remove_radius_outlier(nb_points=10, radius=0.01)
        xyz = np.asarray(pcd.points)
        rgb_flatten = np.asarray(pcd.colors)
        combined_xyz.append(xyz)
        combined_rgb.append(rgb_flatten)

        xyz = make_pointcloud_from_rgbd(rgb, new_depth, intrinsics, extrinsics)
        xyz = xyz[mask]
        rgb_flatten = rgb.reshape(-1, 3)[mask] / 255
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb_flatten)
        pcd, indices = pcd.remove_radius_outlier(nb_points=10, radius=0.01)
        xyz = np.asarray(pcd.points)
        rgb_flatten = np.asarray(pcd.colors)
        new_combined_xyz.append(xyz)
        new_combined_rgb.append(rgb_flatten)

    combined_xyz = np.concatenate(combined_xyz, axis=0)
    combined_rgb = np.concatenate(combined_rgb, axis=0)
    new_combined_xyz = np.concatenate(new_combined_xyz, axis=0)
    new_combined_rgb = np.concatenate(new_combined_rgb, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_xyz)
    pcd.colors = o3d.utility.Vector3dVector(combined_rgb)

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_combined_xyz)
    new_pcd.colors = o3d.utility.Vector3dVector(new_combined_rgb)

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([pcd, coordinate])
    o3d.visualization.draw_geometries([new_pcd, coordinate])
    o3d.visualization.draw_geometries([pcd, new_pcd, coordinate])


if __name__ == '__main__':
    main()
