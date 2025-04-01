#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : camera.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/25/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import math
import collections
import numpy as np
import pybullet as p

from typing import Optional, Union, Tuple
from concepts.math.rotationlib_xyzw import rpy

__all__ = [
    'CameraConfig', 'RealSenseD415', 'TopDownOracle', 'RS200Gazebo', 'KinectFranka', 'CLIPortCamera',
    'get_point_cloud', 'get_point_cloud_image', 'get_orthographic_heightmap', 'lookat_rpy',
    'kabsch_transform', 'SimpleCameraTransform'
]


class CameraConfig(collections.namedtuple('_CameraConfig', ['image_size', 'intrinsics', 'position', 'rotation', 'zrange', 'shadow'])):
    """
    Mostly based on https://github.com/cliport/cliport/blob/e9cde74754606448d8a0495c1efea36c29b201f1/cliport/tasks/cameras.py
    """

    def get_view_and_projection_matricies(self, image_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        if image_size is None:
            image_size = self.image_size

        view_matrix = self.get_view_matrix()
        projection_matrix = self.get_projection_matrix(image_size)
        return view_matrix, projection_matrix

    def get_view_matrix(self) -> np.ndarray:
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(rpy(*self.rotation))
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = self.position + lookdir
        return p.computeViewMatrix(self.position, lookat, updir)

    def get_projection_matrix(self, image_size: Tuple[int, int]) -> np.ndarray:
        focal_len = self.intrinsics[0]
        znear, zfar = self.zrange
        fovh = (image_size[0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = float(image_size[1]) / image_size[0]
        return p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

    def get_intrinsics_matrix(self) -> np.ndarray:
        return np.array(self.intrinsics).reshape(3, 3)

    def get_extrinsics_matrix(self) -> np.ndarray:
        rotation = p.getMatrixFromQuaternion(rpy(*self.rotation))
        rotm = np.float32(rotation).reshape(3, 3)
        position = np.array(self.position)
        transform = np.eye(4)
        transform[:3, :3] = rotm
        transform[:3, 3] = position
        return transform


class RealSenseD415(object):
    """Default configuration with 3 RealSense RGB-D cameras."""

    # Mimic RealSense D415 RGB-D camera parameters.
    image_size = (480, 640)
    intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)

    # Set default camera poses.
    front_position = (1., 0, 0.75)
    front_rotation = np.rad2deg((np.pi / 4, np.pi, -np.pi / 2))
    left_position = (0, 0.5, 0.75)
    left_rotation = np.rad2deg((np.pi / 4.5, np.pi, np.pi / 4))
    right_position = (0, -0.5, 0.75)
    right_rotation = np.rad2deg((np.pi / 4.5, np.pi, 3 * np.pi / 4))

    @classmethod
    def get_configs(cls):
        return [
            CameraConfig(cls.image_size, cls.intrinsics, cls.front_position, cls.front_rotation, (0.1, 10.), True),
            CameraConfig(cls.image_size, cls.intrinsics, cls.left_position, cls.left_rotation, (0.1, 10.), True),
            CameraConfig(cls.image_size, cls.intrinsics, cls.right_position, cls.right_rotation, (0.1, 10.), True)
        ]


class TopDownOracle(object):
    """Top-down noiseless image used only by the oracle demonstrator."""

    # Near-orthographic projection.
    image_size = (480, 640)
    intrinsics = (63e4, 0, 320., 0, 63e4, 240., 0, 0, 1)
    position = (0.5, 0, 1000.)
    rotation = np.rad2deg((0, np.pi, -np.pi / 2))

    @classmethod
    def get_configs(cls):
        return [CameraConfig(cls.image_size, cls.intrinsics, cls.position, cls.rotation, (999.7, 1001.), False)]


class RS200Gazebo(object):
    """Gazebo Camera"""

    # Near-orthographic projection.
    image_size = (480, 640)
    intrinsics = (554.3826904296875, 0.0, 320.0, 0.0, 554.3826904296875, 240.0, 0.0, 0.0, 1.0)
    position = (0.5, 0, 1.0)
    rotation = np.rad2deg((0, np.pi, np.pi / 2))

    @classmethod
    def get_configs(cls):
        return [CameraConfig(cls.image_size, cls.intrinsics, cls.position, cls.rotation, (0.1, 10.), False)]


class KinectFranka(object):
    """Kinect Franka Camera"""

    # Near-orthographic projection.
    image_size = (424, 512)
    intrinsics = (365.57489013671875, 0.0, 257.5205078125, 0.0, 365.57489013671875, 205.26710510253906, 0.0, 0.0, 1.0)
    position = (1.082, -0.041, 1.027)
    rotation = np.rad2deg((-2.611, 0.010, 1.553))

    @classmethod
    def get_configs(cls):
        return [CameraConfig(cls.image_size, cls.intrinsics, cls.position, cls.rotation, (0.1, 10.), True)]


class CLIPortCamera(object):
    """CLIPort camera configuration."""

    image_size = (480, 640)
    intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)
    position1 = (1.5, 0, 0.8)
    rotation1 = (0, -115, 90)
    position2 = (0, -1, 0.8)
    rotation2 = (-120, 20, 10)
    position3 = (0, 1, 0.8)
    rotation3 = (120, 20, 170)

    @classmethod
    def get_configs(cls):
        return [
            CameraConfig(cls.image_size, cls.intrinsics, cls.position1, cls.rotation1, (0.1, 10.), False),
            CameraConfig(cls.image_size, cls.intrinsics, cls.position2, cls.rotation2, (0.1, 10.), False),
            CameraConfig(cls.image_size, cls.intrinsics, cls.position3, cls.rotation3, (0.1, 10.), False)
        ]


def get_point_cloud(config, color, depth, segmentation=None):
    """Reconstruct point clouds from the color and depth images."""
    # based on https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer
    height, width = depth.shape[0:2]
    view_matrix, proj_matrix = config.get_view_and_projection_matricies()

    # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

    # create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
    y *= -1.
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    mask = z < 0.99
    # mask = ...
    # filter out "infinite" depths
    pixels = pixels[mask]
    pixels[:, 2] = 2 * pixels[:, 2] - 1

    # turn pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3: 4]
    points = points[:, :3]

    if segmentation is not None:
        return points, color.reshape(-1, color.shape[-1])[mask], segmentation.reshape(-1)[mask]
    return points, color.reshape(-1, color.shape[-1])[mask]


def get_point_cloud_image(config: CameraConfig, depth: np.ndarray) -> np.ndarray:
    """Reconstruct point clouds from the depth image. This function differs from the :func:`get_point_cloud` function in that it returns a point cloud "image",
    that is, a 3D point for each pixel in the depth image.

    Args:
        config: camera configuration.
        depth: depth image, of shape (H, W).

    Returns:
        points: point cloud image, of shape (H, W, 3).
    """
    intrinsics = np.array(config.intrinsics).reshape(3, 3)

    height, width = depth.shape[0:2]
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)

    position = np.array(config.position).reshape(3, 1)
    rotation = np.array(p.getMatrixFromQuaternion(rpy(*config.rotation))).reshape(3, 3)
    transform = np.eye(4)
    transform[:3, :] = np.hstack((rotation, position))

    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(
        points.copy(), padding,
        'constant', constant_values=1
    )
    for i in range(3):
        points[..., i] = np.sum(transform[i, :] * homogen_points, axis=-1)

    return points


def get_orthographic_heightmap(pcd_image: np.ndarray, color_image: np.ndarray, bounds: np.ndarray, pixel_size: float, segmentation: Optional[np.ndarray] = None) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Get a top-down (z-axis) orthographic heightmap image from 3D pointcloud.

    Args:
        pcd_image: HxWx3 float array of 3D points in world coordinates.
        color_image: HxWx3 uint8 array of values in range 0-255 aligned with points.
        bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining the region in 3D space to generate heightmap in world coordinates.
        pixel_size: float defining size of each pixel in meters.
        segmentation: HxW int32 array of segmentation mask aligned with points.

    Returns:
        heightmap: HxW float array of height (from lower z-bound) in meters.
        colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
        segmentation: HxW int32 array of backprojected segmentation mask aligned with heightmap.
    """
    if segmentation is not None:
        color_image = np.concatenate((color_image, segmentation[..., None]), axis=2)

    width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
    height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
    heightmap = np.zeros((height, width), dtype=np.float32)
    colormap = np.zeros((height, width, color_image.shape[-1]), dtype=np.uint8)

    # Filter out 3D points that are outside of the predefined bounds.
    ix = (pcd_image[Ellipsis, 0] >= bounds[0, 0]) & (pcd_image[Ellipsis, 0] < bounds[0, 1])
    iy = (pcd_image[Ellipsis, 1] >= bounds[1, 0]) & (pcd_image[Ellipsis, 1] < bounds[1, 1])
    iz = (pcd_image[Ellipsis, 2] >= bounds[2, 0]) & (pcd_image[Ellipsis, 2] < bounds[2, 1])
    valid = ix & iy & iz
    pcd_image = pcd_image[valid]
    color_image = color_image[valid]

    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(pcd_image[:, -1])
    pcd_image, color_image = pcd_image[iz], color_image[iz]
    px = np.int32(np.floor((pcd_image[:, 0] - bounds[0, 0]) / pixel_size))
    py = np.int32(np.floor((pcd_image[:, 1] - bounds[1, 0]) / pixel_size))
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    heightmap[py, px] = pcd_image[:, 2] - bounds[2, 0]
    for c in range(color_image.shape[-1]):
        colormap[py, px, c] = color_image[:, c]

    if segmentation is not None:
        return heightmap, colormap[..., :3], colormap[..., 3]
    return heightmap, colormap


def lookat_rpy(camera_pos: np.ndarray, target_pos: np.ndarray, roll: float = 0) -> np.ndarray:
    """Construct the roll, pitch, yaw angles of a camera looking at a target.
    This function assumes that the camera is pointing to the z-axis ([0, 0, 1]),
    in the camera frame.

    Args:
        camera_pos: the position of the camera.
        target_pos: the target position.
        roll: the roll angle of the camera.

    Returns:
        a numpy array of the roll, pitch, yaw angles in degrees.
    """
    camera_pos = np.asarray(camera_pos)
    target_pos = np.asarray(target_pos)

    delta = target_pos - camera_pos
    pitch = math.atan2(-np.linalg.norm(delta[:2]), delta[2])
    yaw = math.atan2(-delta[1], -delta[0])
    rv = np.array([np.deg2rad(roll), pitch, yaw], dtype="float32")

    return np.rad2deg(rv)


def kabsch_transform(x, y, verbose: bool = False):
    """Find optimal rotation and translation to map vectors in x to y

    Args:
        x: ndarray of size (N, 3).
        y: ndarray of size (N, 3).
        verbose: whether to print the intermediate results.

    Returns:
        rotation: ndarray of size (3, 3).
        translation: ndarray of size (3,).
    """

    # Calculate centroids
    centroid_x = np.mean(x, axis=0)
    centroid_y = np.mean(y, axis=0)

    # Centre the vectors
    x_centred = x - centroid_x
    y_centred = y - centroid_y

    # Calculate covariance matrix
    H = np.dot(x_centred.T, y_centred)

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

    if verbose:
        print('kabsch_transform::SVD')
        print(U, S, Vt)

    # Calculate rotation
    d = (np.linalg.det(U) * np.linalg.det(Vt)) < 0.0
    if d:
        S[-1] = -S[-1]
        U[:, -1] = -U[:, -1]

    rotation = np.dot(Vt.T, U.T)

    # Calculate translation
    translation = centroid_y - np.dot(rotation, centroid_x)

    if verbose:
        print('kabsch_transform::Rotation:')
        print(rotation)
        print('kabsch_transform::Translation:')
        print(translation)

    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


class SimpleCameraTransform(object):
    def __init__(self, intrinsics, extrinsics):
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    def w2c(self, x, y, z):
        x, y, z, _ = np.dot(self.extrinsics, np.array([x, y, z, 1]))
        u, v, z = np.dot(self.intrinsics, np.array([x, y, z]))
        u, v = int(u / z), int(v / z)
        return u, v

    def c2w(u, v, d):
        x, y, z = np.dot(np.linalg.inv(self.intrinsics), np.array([u * d, v * d, d]))
        x, y, z, _ = np.dot(np.linalg.inv(self.extrinsics), np.array([x, y, z, 1]))
        return x, y, z
