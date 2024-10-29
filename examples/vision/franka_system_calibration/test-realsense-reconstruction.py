#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test-realsense-reconstruction.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/01/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import cv2
import numpy as np
import open3d as o3d
import jacinle
from concepts.vision.george_vision_pipeline.object_centric_vision import compute_transformation_from_plane_equation


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
    return points.T, rgb.reshape(-1, 3)


def make_open3d_pointcloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def filter_plane(pcd: o3d.geometry.PointCloud, z_threshold=0.1):
    points = np.asarray(pcd.points)
    cond = np.abs(points[:, 2]) < z_threshold
    pcd = pcd.select_by_index(np.where(cond)[0])

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    print(plane_model)
    return plane_model


def make_open3d_plane_object(plane_model):
    table_plane = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=0.001).translate(np.array([0, -0.5, 0]))
    table_plane.paint_uniform_color([0.5, 0.5, 0.5])
    T = compute_transformation_from_plane_equation(*plane_model)
    table_plane = table_plane.transform(T)
    return table_plane


def main():
    cal1 = jacinle.load('./realsense_data/calibration-1.pkl')
    cal2 = jacinle.load('./realsense_data/calibration-2.pkl')

    scene1 = jacinle.load('./realsense_data/scene-1.pkl')
    scene2 = jacinle.load('./realsense_data/scene-2.pkl')

    world1_to_obj = cal1['poses'][0]
    world2_to_obj = cal2['poses'][0]
    world1_to_cam1 = cal1['extrinsics']
    world2_to_cam2 = cal2['extrinsics']

    world2_to_cam1 = world2_to_obj @ np.linalg.inv(world1_to_obj) @ world1_to_cam1

    intrinsics1 = cal1['intrinsics']
    intrinsics2 = cal2['intrinsics']

    pcd1, rgb1 = make_pointcloud_from_rgbd(scene1['color'], scene1['depth'] / 1000, intrinsics1, np.linalg.inv(world2_to_cam1))
    pcd2, rgb2 = make_pointcloud_from_rgbd(scene2['color'], scene2['depth'] / 1000, intrinsics2, np.linalg.inv(world2_to_cam2))

    open3d_pcd1 = make_open3d_pointcloud(pcd1, rgb1)
    open3d_pcd2 = make_open3d_pointcloud(pcd2, rgb2)

    plane1 = make_open3d_plane_object(filter_plane(open3d_pcd1))
    plane2 = make_open3d_plane_object(filter_plane(open3d_pcd2))

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pcd1)
    # pcd.colors = o3d.utility.Vector3dVector(rgb1 / 255)
    # o3d.visualization.draw_geometries([pcd])

    # return

    combined_pcd = np.vstack([pcd1, pcd2])
    combined_rgb = np.vstack([rgb1, rgb2]) / 255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_pcd)
    pcd.colors = o3d.utility.Vector3dVector(combined_rgb)
    o3d.visualization.draw_geometries([pcd, plane1, plane2])


if __name__ == '__main__':
    main()
