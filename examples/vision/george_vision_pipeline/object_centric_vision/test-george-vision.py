#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_george_vision.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/05/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import jacinle
import open3d as o3d
import numpy as np
import cv2
import os.path as osp

from jacinle.comm.service import SocketClient
from concepts.hw_interface.franka.fri_server import FrankaService
from concepts.vision.george_vision_pipeline.object_centric_vision import get_pointcloud, filter_pointcloud_range, ransac_table_detection, CameraTransformation
from concepts.vision.george_vision_pipeline.object_centric_vision import ObjectCentricVisionPipeline
from concepts.vision.george_vision_pipeline.segmentation_models import ImageBasedPCDSegmentationModel, PointGuidedImageSegmentationModel

parser = jacinle.JacArgumentParser()
parser.add_argument('--remote', action='store_true')
args = parser.parse_args()


def main():
    if not args.remote:
        pickled_file = jacinle.load('data/data-000001.pkl')
        jacinle.stprint(pickled_file)
        color_image = pickled_file['color_image'][..., ::-1]  # BGR -> RGB
        depth_image = pickled_file['depth_image']
        print(f'Image loaded. color_image.shape={color_image.shape}, depth_image.shape={depth_image.shape}')
    else:
        host = jacinle.jac_getenv('PANDA_HOST', '128.31.39.162')
        franka = SocketClient('franka', ['tcp://{}:{}'.format(host, FrankaService.DEFAULT_PORTS[0]), 'tcp://{}:{}'.format(host, FrankaService.DEFAULT_PORTS[1])])
        franka.initialize()
        color_image, depth_image = franka.capture_image()
        color_image = color_image[..., ::-1]  # BGR -> RGB
        print('')
        print(f'Image captured. color_image.shape={color_image.shape}, depth_image.shape={depth_image.shape}')

    cv2.imshow('color_image', color_image[..., ::-1])
    key = cv2.waitKey(0)
    if key == ord('q'):
        print('User quit.')
        return

    print('Loading camer transformation... ', end='')
    camera_transformation = CameraTransformation.from_intrinsics_and_extrinsics_file('./calibration/intrinsics.pkl', './calibration/extrinsics.pkl')
    print('Done.')
    print('Loading segmentation model... ', end='')
    segmentation_model = ImageBasedPCDSegmentationModel()
    print('Done.')
    print('Loading SAM model... ', end='')
    sam_model = PointGuidedImageSegmentationModel(osp.expanduser('~/Workspace/datasets/SegmentAnything/sam_vit_h_4b8939.pth'))
    print('Done.')

    oov = ObjectCentricVisionPipeline(color_image, depth_image)
    oov.construct_pointcloud(camera_transformation)
    oov.construct_foreground_by_cropping(x_range=(0.2, 1.0), y_range=(-0.5, 0.5), z_range=(-0.1, 0.5), depth_range=(0.05, 2.0))
    oov.construct_table_segmentation(method='ransac', distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    oov.construct_object_segmentation(segmentation_model, dbscan_with_sam=True, sam_model=sam_model)
    oov.construct_object_completion_by_extrusion()
    oov.construct_object_meshes()

    oov.visualize_pcd('full')
    oov.visualize_pcd('foreground')
    oov.visualize_pcd('table')
    oov.visualize_pcd('tabletop')
    oov.visualize_objects_3d('raw_pcd')
    oov.visualize_objects_3d()
    oov.visualize_objects_2d()
    # oov.visualize_objects_3d('mesh', overlay_background_pcd=True)

    if args.remote:
        latest_dirname = jacinle.io.locate_newest_file('./data', 'scene-*.scene')
        if latest_dirname is None:
            index = 0
        else:
            index = int(latest_dirname.split('-')[-1].split('.')[0])
        this_dirname = f'./data/scene-{index + 1:06d}.scene'
        oov.export_scene(this_dirname)
        print(f'Scene saved to {this_dirname}.')


def main_test_code():
    pickled_file = jacinle.load('data/data-000001.pkl')
    jacinle.stprint(pickled_file)
    color_image = pickled_file['color_image'][..., ::-1]  # BGR -> RGB
    depth_image = pickled_file['depth_image']
    print(f'Image loaded. color_image.shape={color_image.shape}, depth_image.shape={depth_image.shape}')

    camera_transformation = CameraTransformation.from_intrinsics_and_extrinsics_file('./calibration/intrinsics.pkl', './calibration/extrinsics.pkl')
    pcd = get_pointcloud(color_image, depth_image, camera_transformation=camera_transformation)
    # NB(Jiayuan Mao @ 2023/08/05): camera Z is pointed towards the scene, rendered as a blue arrow.
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05).transform(np.linalg.inv(camera_transformation.extrinsics))

    pcd = filter_pointcloud_range(pcd, x_range=(0.1, 0.6), y_range=(-0.6, 0.6), z_range=(-0.1, 1.1))
    # o3d.visualization.draw_geometries([pcd, camera, world_frame])

    plane_pcd, objects_pcd = ransac_table_detection(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    plane_pcd.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([plane_pcd, camera, world_frame])
    # o3d.visualization.draw_geometries([objects_pcd, camera, world_frame])

    # result = segmentation_model.segment_image(color_image)
    # visualize_instance_segmentation(color_image, result)


if __name__ == '__main__':
    main()

