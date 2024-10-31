#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : calibration.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/11/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
from typing import Optional, Tuple, Dict

from concepts.math.frame_utils_xyzw import compose_transformation, calc_transformation_matrix_from_plane_equation
from concepts.math.rotationlib_xyzw import quat2mat
from concepts.hw_interface.realsense.device_f import CaptureRS
from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.pybullet.components.panda.panda_robot import PandaRobot
from concepts.vision.franka_system_calibration.ar_detection import get_ar_tag_detections, get_ar_tag_poses_from_camera_pose

__all__ = [
    'get_camera_pose_using_pybullet', 'get_camera_pose_using_ikfast',
    'get_mounted_camera_pose_from_qpos',
    'get_mounted_camera_pose_from_ar_detections',
    'DepthRangeFilter', 'XYZRangeFilter',
    'make_pointcloud_from_rgbd', 'make_open3d_pointcloud',
    'filter_plane', 'make_open3d_plane_object', 'visualize_calibrated_pointclouds',
    'get_camera_configs_using_ar_detection_from_camera_images', 'get_camera_configs_using_ar_detection',
    'get_world_coordinate_pointclouds'
]


def get_camera_pose_using_pybullet(robot, qpos, hand_to_camera_pos, hand_to_camera_quat):
    ee_pos, ee_quat = robot.fk(qpos, 8)

    pos, quat = compose_transformation(ee_pos, ee_quat, hand_to_camera_pos, hand_to_camera_quat)

    camera_pose = np.eye(4)
    camera_pose[:3, :3] = quat2mat(quat)
    camera_pose[:3, 3] = pos
    return camera_pose


def get_camera_pose_using_ikfast(qpos, hand_to_camera_pos, hand_to_camera_quat):
    from concepts.simulator.ikfast.quickaccess import get_franka_panda_ikfast, franka_panda_fk

    pos, quat = franka_panda_fk(get_franka_panda_ikfast(), qpos)
    pos, quat = compose_transformation(pos, quat, hand_to_camera_pos, hand_to_camera_quat)

    camera_pose = np.eye(4)
    camera_pose[:3, :3] = quat2mat(quat)
    camera_pose[:3, 3] = pos
    return camera_pose


def get_mounted_camera_pose_from_qpos(qpos, hand_to_camera_pos, hand_to_camera_quat, fk_method: str = 'ikfast'):
    if fk_method == 'pybullet':
        client = BulletClient(is_gui=False)
        panda_robot = PandaRobot(client)
        camera_pose = get_camera_pose_using_pybullet(panda_robot, qpos, hand_to_camera_pos, hand_to_camera_quat)
        client.disconnect()
    elif fk_method == 'ikfast':
        camera_pose = get_camera_pose_using_ikfast(qpos, hand_to_camera_pos, hand_to_camera_quat)
    else:
        raise ValueError(f'Unsupported fk method: {fk_method}')
    return camera_pose


def get_mounted_camera_pose_from_ar_detections(hand_camera_ar_poses, mounted_camera_ar_poses):
    for key in hand_camera_ar_poses.keys():
        if key not in mounted_camera_ar_poses.keys():
            continue

        hand_camera_ar_pose = hand_camera_ar_poses[key]
        mounted_camera_ar_pose = mounted_camera_ar_poses[key]
        mounted_camera_in_base = hand_camera_ar_pose @ np.linalg.inv(mounted_camera_ar_pose)
        return mounted_camera_in_base

    raise ValueError("No common AR tag found between hand camera and mounted camera")


class DepthRangeFilter(object):
    def __init__(self, min_depth, max_depth = None):
        self.min_depth = min_depth if max_depth is not None else 0
        self.max_depth = max_depth if max_depth is not None else min_depth

    def __call__(self, depth):
        return np.logical_and(self.min_depth < depth, depth <= self.max_depth)


class XYZRangeFilter(object):
    def __init__(self, x_range, y_range, z_range):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

    def __call__(self, x, y, z):
        return np.logical_and.reduce([
            self.x_range[0] < x, x < self.x_range[1],
            self.y_range[0] < y, y < self.y_range[1],
            self.z_range[0] < z, z < self.z_range[1]
        ])


def make_pointcloud_from_rgbd(rgb, depth, intrinsics, extrinsics, normalize_depth=True, depth_filter_fn=None, xyz_filter_fn=None):
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.flatten()
    y = y.flatten()
    z = depth.flatten() / 1000 if normalize_depth else depth.flatten()
    rgb = rgb.reshape(-1, 3)

    if depth_filter_fn is not None:
        mask = depth_filter_fn(z)
        x, y, z = x[mask], y[mask], z[mask]
        rgb = rgb[mask]

    points = np.vstack([x, y, np.ones_like(x)])
    points = (np.linalg.inv(intrinsics) @ points) * z
    points = np.vstack([points, np.ones(points.shape[1])])
    points = extrinsics @ points
    points = points[:3] / points[3]
    points = points.T

    if xyz_filter_fn is not None:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        mask = xyz_filter_fn(x, y, z)
        points = points[mask]
        rgb = rgb[mask]
    return points, rgb


def make_open3d_pointcloud(points, colors):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255)
    return pcd


def filter_plane(pcd, z_threshold=0.1):
    points = np.asarray(pcd.points)
    cond = np.abs(points[:, 2]) < z_threshold
    pcd = pcd.select_by_index(np.where(cond)[0])

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    return plane_model


def make_open3d_plane_object(plane_model):
    import open3d as o3d
    table_plane = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=0.001).translate(np.array([0, 0, 0]))
    table_plane.paint_uniform_color([0.5, 0.5, 0.5])
    T = calc_transformation_matrix_from_plane_equation(*plane_model)
    table_plane = table_plane.transform(T)
    return table_plane


def visualize_calibrated_pointclouds(
    camera_configs, rgbd_images, depth_filter_fns=None,
    xyz_filter_fn=None,
):
    import open3d as o3d
    geometries = []
    for camera_name, rgbd_image in rgbd_images.items():
        points, colors = make_pointcloud_from_rgbd(
            rgbd_image[0], rgbd_image[1],
            camera_configs[camera_name]['intrinsics'], camera_configs[camera_name]['extrinsics'],
            depth_filter_fn=depth_filter_fns.get(camera_name, None) if depth_filter_fns is not None else None,
            xyz_filter_fn=xyz_filter_fn
        )
        pcd = make_open3d_pointcloud(points, colors)
        geometries.append(pcd)
        plane_model = filter_plane(pcd)
        table_plane = make_open3d_plane_object(plane_model)
        geometries.append(table_plane)

    o3d.visualization.draw_geometries(geometries)


def get_camera_configs_using_ar_detection_from_camera_images(
    camera_images: Dict[str, Tuple[np.ndarray, np.ndarray]],
    camera_intrinsics: Dict[str, np.ndarray],
    reference_camera_name: str, reference_camera_pose: np.ndarray,
    calibrate_robot: bool = False,
    robot_to_camera_transforms: Optional[Dict[str, Tuple[str, np.ndarray]]] = None
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    camera_configs = {k: {} for k in camera_images.keys()}
    camera_ar_tag_poses = dict()
    for camera_name, (color, depth) in camera_images.items():
        ar_detections = get_ar_tag_detections(color)
        if len(ar_detections) != 1:
            raise ValueError(f"AR tag detection failed: detected {len(ar_detections)} tags")
        if camera_name == reference_camera_name:
            ar_poses = get_ar_tag_poses_from_camera_pose(ar_detections, camera_intrinsics[reference_camera_name], reference_camera_pose)
        else:
            ar_poses = get_ar_tag_poses_from_camera_pose(ar_detections, camera_intrinsics[camera_name], np.eye(4))
        camera_ar_tag_poses[camera_name] = ar_poses

    for camera_name in camera_images.keys():
        ar_tag_poses = camera_ar_tag_poses[camera_name]
        if camera_name == reference_camera_name:
            camera_configs[camera_name]['intrinsics'] = camera_intrinsics[reference_camera_name]
            camera_configs[camera_name]['extrinsics'] = reference_camera_pose
        else:
            camera_configs[camera_name]['intrinsics'] = camera_intrinsics[camera_name]
            camera_configs[camera_name]['extrinsics'] = get_mounted_camera_pose_from_ar_detections(camera_ar_tag_poses[reference_camera_name], ar_tag_poses)

    if calibrate_robot:
        assert robot_to_camera_transforms is not None
        for robot_name, (camera_name, base_to_camera_transform) in robot_to_camera_transforms.items():
            camera_pose = camera_configs[camera_name]['extrinsics']
            camera_configs[robot_name] = {'extrinsics': camera_pose @ np.linalg.inv(base_to_camera_transform)}

    return camera_configs, camera_images


def get_camera_configs_using_ar_detection(
    cameras: Dict[str, CaptureRS],
    reference_camera_name: str, reference_camera_pose: np.ndarray,
    calibrate_robot: bool = False,
    robot_to_camera_transforms: Optional[Dict[str, Tuple[str, np.ndarray]]] = None
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    camera_images = dict()
    camera_intrinsics = dict()
    for camera_name, camera in cameras.items():
        color, depth = camera.capture()
        camera_images[camera_name] = (color, depth)
        camera_intrinsics[camera_name] = camera.intrinsics[0]

    return get_camera_configs_using_ar_detection_from_camera_images(camera_images, camera_intrinsics, reference_camera_name, reference_camera_pose, calibrate_robot, robot_to_camera_transforms)


def get_world_coordinate_pointclouds(camera_configs, rgbd_images) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    pointclouds = dict()
    for camera_name, rgbd_image in rgbd_images.items():
        points, colors = make_pointcloud_from_rgbd(rgbd_image[0], rgbd_image[1], camera_configs[camera_name]['intrinsics'], camera_configs[camera_name]['extrinsics'])
        pointclouds[camera_name] = (points, colors)
    return pointclouds


def get_world_coordinate_pointclouds_v2(camera_captures, depth_filter_fns=None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    pointclouds = dict()
    for camera_name, camera_capture in camera_captures.items():
        points, colors = make_pointcloud_from_rgbd(camera_capture['color'], camera_capture['depth'], camera_capture['intrinsics'], camera_capture['extrinsics'])

        if depth_filter_fns is not None:
            if camera_name in depth_filter_fns:
                mask = depth_filter_fns[camera_name](camera_capture['depth'].flatten())
                points, colors = points[mask], colors[mask]
            elif '*' in depth_filter_fns:
                mask = depth_filter_fns['*'](camera_capture['depth'].flatten())
                points, colors = points[mask], colors[mask]
        pointclouds[camera_name] = (points, colors)
    return pointclouds
