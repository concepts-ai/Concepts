#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : run-uncos.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/31/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp
import jacinle
import cv2
import numpy as np
import torch
import open3d as o3d

from concepts.vision.uncos.uncos import UncOS, UncOSConfig

def get_qr_grounding_dino_ckpt_path():
    import platformdirs
    import os.path as osp

    return osp.join(platformdirs.user_cache_dir('QR'), 'groundingdino_swint_ogc.pth')


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
    data = jacinle.load('../_assets/realsense-hook-test.pkl')

    camera_index = 1  # Use the second camera.
    this_camera = data['cameras'][camera_index]
    rgb = cv2.imdecode(this_camera['rgb'], cv2.IMREAD_COLOR)
    depth = cv2.imdecode(this_camera['depth'], cv2.IMREAD_UNCHANGED) / 1000.

    intrinsics = this_camera['intrinsics']
    extrinsics = this_camera['extrinsics']

    pcd = make_pointcloud_from_rgbd(rgb, depth, intrinsics, np.eye(4))
    pcd = pcd.reshape(rgb.shape[0], rgb.shape[1], 3)

    print('Visualizing point cloud...')
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.reshape(-1, 3))
    o3d_pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3) / 255.)
    o3d.visualization.draw_geometries([o3d_pcd])

    print('Loading UncOS...')
    sam_ckpt_path = './sam_vit_h_4b8939.pth'
    grounding_dino_ckpt_path = get_qr_grounding_dino_ckpt_path()

    if not osp.isfile(sam_ckpt_path):
        rv = jacinle.yes_or_no(f'SAM checkpoint not found. Do you want to download it to {sam_ckpt_path}? Enter Yes to continue and download or No to exit.', default='no')
        if not rv:
            return
    if not osp.isfile(grounding_dino_ckpt_path):
        rv = jacinle.yes_or_no(f'GroundingDINO checkpoint not found. Do you want to download it to {grounding_dino_ckpt_path}? Enter Yes to continue and download or No to exit.', default='no')
        if not rv:
            return

    uncos = UncOS(
        UncOSConfig.make_default(max_depth=1.2),
        sam_ckpt_path=sam_ckpt_path, grounding_dino_ckpt_path=grounding_dino_ckpt_path
    )

    print('Running UncOS...')
    uncos.set_image(rgb)
    table_mask = uncos.get_table_or_background_mask(pcd, far=1.2, include_background=True, dbscan_refine=False)

    print('Table mask area:', np.sum(table_mask), '/', table_mask.size)

    print('Visualizing table mask...')
    table_mask_visualize = (table_mask > 0).flatten()
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.reshape(-1, 3)[table_mask_visualize])
    o3d_pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3)[table_mask_visualize] / 255.)
    o3d.visualization.draw_geometries([o3d_pcd])

    test_most_likely = False
    pred_masks_boolarray, uncertain_hypotheses = uncos.segment_scene(
        rgb, pcd, table_or_background_mask=table_mask,
        return_most_likely_only=test_most_likely, n_seg_hypotheses_trial=12
    )
    if test_most_likely:
        assert len(uncertain_hypotheses) == 0
    uncos.visualize_confident_uncertain(pred_masks_boolarray, uncertain_hypotheses)


if __name__ == '__main__':
    main()


