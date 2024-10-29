#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test-depth-anything.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/30/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import jacinle
import os.path as osp
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from transformers import pipeline
from PIL import Image


def load_rgbd(base_path, index):
    rgb = Image.open(osp.join(base_path, str(index), 'rgb_0.png'))
    rgb = np.asarray(rgb)[..., :3][..., ::-1]
    rgb = np.ascontiguousarray(rgb)
    depth = np.load(osp.join(base_path, str(index), 'depth_0.npy'))
    return rgb, depth / 1000

def load_intrinsics_and_extrinsics(base_path):
    intrinsics = np.load(osp.join(base_path, 'intrinsics_0.npy'))
    extrinsics = np.load(osp.join(base_path, 'extrinsics_0.pkl'), allow_pickle=True)
    return intrinsics, extrinsics


def main():
    dam_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    base_dir = '/Users/jiayuanm/Downloads/double_camera_hook/scooper'
    rgb, depth = load_rgbd(base_dir, 1)

    predicted_depth_rv = dam_pipe(Image.fromarray(rgb))

    predicted_depth = predicted_depth_rv['predicted_depth']
    predicted_depth = predicted_depth[0]

    # target_shape = depth.shape
    target_shape = predicted_depth.shape[-2:]

    predicted_depth = F.interpolate(predicted_depth.reshape(1, 1, *predicted_depth.shape), size=target_shape, mode='nearest', align_corners=None)[0, 0]
    predicted_depth = predicted_depth.squeeze().cpu().numpy()
    rgb = F.interpolate(torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0), size=target_shape, mode='bilinear', align_corners=False)[0].permute(1, 2, 0).numpy()
    depth = F.interpolate(torch.tensor(depth).unsqueeze(0).unsqueeze(0), size=target_shape, mode='nearest', align_corners=None)[0, 0].numpy()

    # Fit an affine transformation
    predicted_depth = fit_depth_affine(depth, predicted_depth)

    # Load the camera intrinsics and extrinsics
    intrinsics, extrinsics = load_intrinsics_and_extrinsics(base_dir)

    # Create a point cloud
    points = make_pointcloud_from_rgbd(rgb, predicted_depth, intrinsics, extrinsics)

    # Visualize the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb.reshape([-1, 3]) / 255)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([pcd, coord])


def fit_depth_affine(depth, predicted_depth):
    # fit k, b for depth = k * predicted_depth + b
    depth_flat = depth.flatten()
    predicted_depth_flat = predicted_depth.flatten()

    mask = depth_flat > 0
    depth_flat = depth_flat[mask]
    predicted_depth_flat = predicted_depth_flat[mask]

    A = np.vstack([predicted_depth_flat, np.ones(len(predicted_depth_flat))]).T
    k, b = np.linalg.lstsq(A, depth_flat, rcond=None)[0]
    print(f'Fitted depth = {k:.4f} * predicted_depth + {b:.4f}')
    return k * predicted_depth + b


def make_pointcloud_from_rgbd(rgb, depth, intrinsics, extrinsics):
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.flatten()
    y = y.flatten()
    z = depth.flatten()
    points = np.stack([x, y, z], axis=0)

    points = np.linalg.inv(intrinsics) @ points
    points = np.linalg.inv(extrinsics) @ np.vstack([points, np.ones([1, points.shape[1]])])
    return points[:3].T


if __name__ == '__main__':
    main()

