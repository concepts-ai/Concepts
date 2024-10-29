#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test-depth-anything-smoother.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/30/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import jacinle  # To enable jac-debug
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from typing import Sequence
from PIL import Image

from concepts.vision.depth_smoother.depth_anything_smoother import DepthAnythingV2Smoother


def load_rgbd(base_path, index, camera_index=0):
    rgb = Image.open(osp.join(base_path, str(index), f'rgb_{camera_index}.png'))
    rgb = np.asarray(rgb)[..., :3][..., ::-1]
    depth = np.load(osp.join(base_path, str(index), f'depth_{camera_index}.npy'))
    return rgb, depth / 1000


def load_intrinsics_and_extrinsics(base_path, camera_index=0):
    intrinsics = np.load(osp.join(base_path, f'intrinsics_{camera_index}.npy'))
    extrinsics = np.load(osp.join(base_path, f'extrinsics_{camera_index}.pkl'), allow_pickle=True)
    return intrinsics, extrinsics


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


def make_o3d_pointcloud(rgb, depth, intrinsics, extrinsics, outlier_filter_radius=0.01, outlier_filter_nr_points=50):
    points = make_pointcloud_from_rgbd(rgb, depth, intrinsics, extrinsics)

    new_depth_flat = depth.flatten()
    mask = np.logical_and(new_depth_flat > 0, new_depth_flat < 1.2)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[mask])
    pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3)[mask] / 255)
    pcd, _ = pcd.remove_radius_outlier(outlier_filter_nr_points, radius=outlier_filter_radius)
    return pcd


def merge_pcds(all_pcds: Sequence[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
    agg_points = list()
    agg_rgb = list()
    for pcd in all_pcds:
        agg_points.append(np.asarray(pcd.points))
        agg_rgb.append(np.asarray(pcd.colors))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(agg_points, axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(agg_rgb, axis=0))
    return pcd


def smooth(rgb, depth, mask, visualize_fitting: bool = False, visualize_predictions: bool = False, visualize_distributions: bool = False, visualization_title=None):
    new_depth = smoother(rgb, depth, mask, visualize=visualize_fitting)

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

    return new_depth


smoother = DepthAnythingV2Smoother()
coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)


def main_realsense():
    all_pcds = list()
    base_dir = '/Users/jiayuanm/Downloads/double_camera_hook/scooper'

    for j in range(2):
        new_depths = list()
        for i in range(1, 1 + 1):
            rgb, depth = load_rgbd(base_dir, i, j)
            mask = depth < 1.2
            new_depth = smooth(rgb, depth, mask, visualize_fitting=True, visualization_title=f'Camera {j} Frame {i}')
            new_depths.append(new_depth)

        new_depth = np.stack(new_depths, axis=0)
        new_depth = np.mean(new_depth, axis=0)
        intrinsics, extrinsics = load_intrinsics_and_extrinsics(base_dir, j)

        raw_pcd = make_o3d_pointcloud(rgb, depth, intrinsics, extrinsics)
        print('Visualizing the raw point cloud for camera', j)
        o3d.visualization.draw_geometries([raw_pcd, coord])

        pcd = make_o3d_pointcloud(rgb, new_depth, intrinsics, extrinsics)
        print('Visualizing the smoothed point cloud for camera', j)
        o3d.visualization.draw_geometries([pcd, coord])

        print('Visualizing the merged point cloud for camera', j)
        o3d.visualization.draw_geometries([raw_pcd, pcd, coord])

        all_pcds.append(pcd)

    pcd = merge_pcds(all_pcds)
    print('Visualizing the merged point cloud')
    o3d.visualization.draw_geometries([pcd, coord])


def main_realsense_from_saved_pickle():
    data = jacinle.load('../_assets/realsense-hook-test.pkl')

    all_pcds = list()
    for j in range(2):
        this_camera = data['cameras'][j]
        rgb = cv2.imdecode(this_camera['rgb'], cv2.IMREAD_COLOR)
        depth = cv2.imdecode(this_camera['depth'], cv2.IMREAD_UNCHANGED) / 1000.

        intrinsics = this_camera['intrinsics']
        extrinsics = this_camera['extrinsics']

        mask = depth < 1.2
        new_depth = smooth(rgb, depth, mask, visualize_fitting=True, visualization_title=f'Camera {j}')

        raw_pcd = make_o3d_pointcloud(rgb, depth, intrinsics, extrinsics)
        print('Visualizing the raw point cloud for camera', j)
        o3d.visualization.draw_geometries([raw_pcd, coord])

        pcd = make_o3d_pointcloud(rgb, new_depth, intrinsics, extrinsics)
        print('Visualizing the smoothed point cloud for camera', j)
        o3d.visualization.draw_geometries([pcd, coord])

        print('Visualizing the merged point cloud for camera', j)
        o3d.visualization.draw_geometries([raw_pcd, pcd, coord])

        all_pcds.append(pcd)

    pcd = merge_pcds(all_pcds)
    print('Visualizing the merged point cloud')
    o3d.visualization.draw_geometries([pcd, coord])


def main_movo():
    data = jacinle.io.load('../_assets/movo-rgbd-test.pkl')
    rgb = data['rgb']
    depth = data['depth']
    intrinsics = data['intrinsics']
    extrinsics = data['extrinsics']

    new_depth = smooth(rgb, depth, depth < 1.2, visualize_fitting=True, visualization_title='MOVO')

    raw_pcd = make_o3d_pointcloud(rgb, depth, intrinsics, extrinsics)
    print('Visualizing the raw point cloud')
    o3d.visualization.draw_geometries([raw_pcd, coord])

    pcd = make_o3d_pointcloud(rgb, new_depth, intrinsics, extrinsics)
    print('Visualizing the smoothed point cloud')
    o3d.visualization.draw_geometries([pcd, coord])

    print('Visualizing the merged point cloud')
    o3d.visualization.draw_geometries([raw_pcd, pcd, coord])


if __name__ == '__main__':
    # main_realsense()
    main_realsense_from_saved_pickle()
    # main_movo()

