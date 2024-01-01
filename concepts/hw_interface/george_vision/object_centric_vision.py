#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : object_centric_vision.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/05/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp
from typing import Optional, Union, Tuple, List, Dict
from functools import cached_property
from dataclasses import dataclass

import jacinle
import jacinle.io as io
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from concepts.utils.rotationlib import axisangle2quat, quat2mat

from concepts.hw_interface.george_vision.segmentation_models import random_colored_mask, visualize_instance_segmentation, ImageBasedPCDSegmentationModel, PointGuidedImageSegmentationModel, InstanceSegmentationResult
from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.pybullet.urdf_utils.obj2urdf import ObjectUrdfBuilder


__all__ = [
    'CameraTransformation', 'ObjectDetectionResult', 'ObjectCentricVisionPipeline',
    'get_pointcloud', 'filter_pointcloud_range', 'ransac_table_detection', 'threshold_table_detection',
    'project_pointcloud_to_plane', 'object_reconstruction_by_extrusion', 'remove_outliers', 'mesh_reconstruction_alpha_shape',
    'canonize_mesh_center_', 'compute_transformation_from_plane_equation', 'visualize_with_camera_matrix',
    'load_scene_in_pybullet',
]


class CameraTransformation(object):
    """Camera transformation class."""

    def __init__(self, *, intrinsics=None, extrinsics=None):
        if intrinsics is None:
            intrinsics = type(self).default_intrinsics()

        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    intrinsics: np.ndarray
    """Intrinsic matrix of the camera."""

    extrinsics: Optional[np.ndarray]
    """Extrinsic matrix of the camera."""

    @classmethod
    def from_extrinsics_file(cls, filename):
        """Load the camera transformation from an extrinsics file."""
        return cls(intrinsics=None, extrinsics=io.load(filename))

    @classmethod
    def from_intrinsics_and_extrinsics_file(cls, intrinsics_file, extrinsics_file):
        """Load the camera transformation from an intrinsics file and an extrinsics file."""
        return cls(intrinsics=io.load(intrinsics_file), extrinsics=io.load(extrinsics_file))

    @classmethod
    def default_intrinsics(cls):
        """Returns the default intrinsic matrix."""
        intrinsics = np.array([
            [1.36275940e+03, 0.00000000e+00, 9.39736938e+02],
            [0.00000000e+00, 1.36269678e+03, 5.67980225e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])
        return intrinsics

    def w2c(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray], z: Union[float, np.ndarray]):
        """Converts a point in the world coordinate system to the camera coordinate system. The inputs can be either scalars or arrays.

        Args:
            x: x coordinate(s) in the world coordinate system.
            y: y coordinate(s) in the world coordinate system.
            z: z coordinate(s) in the world coordinate system.

        Returns:
            a tuple of (u, v, z) in the camera coordinate system.
        """
        if isinstance(x, float) and isinstance(y, float) and isinstance(z, float):
            x, y, z, _ = np.dot(self.extrinsics, np.array([x, y, z, 1]))
            u, v, z = np.dot(self.intrinsics, np.array([x, y, z]))
            u, v = int(u / z), int(v / z)
            return u, v
        else:
            x, y, z = map(lambda x: np.asarray(x).reshape(-1), [x, y, z])
            xyz = np.stack([x, y, z, np.ones_like(x)], axis=-1)
            xyz = np.dot(self.extrinsics, xyz.transpose(1, 0)).transpose(1, 0)[:, :3]
            uvz = np.dot(self.intrinsics, xyz.transpose(1, 0)).transpose(1, 0)
            u, v, z = uvz[:, 0] / uvz[:, 2], uvz[:, 1] / uvz[:, 2], uvz[:, 2]
            return u.astype(np.int32), v.astype(np.int32), z

    def c2w(self, u: Union[float, np.ndarray], v: Union[float, np.ndarray], d: Union[float, np.ndarray]):
        """Converts a point in the camera coordinate system to the world coordinate system. The inputs can be either scalars or arrays.

        Args:
            u: u coordinate(s) in the camera coordinate system.
            v: v coordinate(s) in the camera coordinate system.
            d: depth(s) in the camera coordinate system.

        Returns:
            a tuple of (x, y, z) in the world coordinate system.
        """
        if isinstance(u, float) and isinstance(v, float) and isinstance(d, float):
            x, y, z = np.dot(np.linalg.inv(self.intrinsics), np.array([u * d, v * d, d]))
            x, y, z, _ = np.dot(np.linalg.inv(self.extrinsics), np.array([x, y, z, 1]))
            return x, y, z
        else:
            u, v, d = map(lambda x: np.asarray(x).reshape(-1), [u, v, d])
            xyz = np.stack([u * d, v * d, d], axis=-1)
            xyz = np.dot(np.linalg.inv(self.intrinsics), xyz.transpose(1, 0)).transpose(1, 0)
            xyz = np.stack([xyz[:, 0], xyz[:, 1], xyz[:, 2], np.ones_like(xyz[:, 0])], axis=-1)
            xyz = np.dot(np.linalg.inv(self.extrinsics), xyz.transpose(1, 0)).transpose(1, 0)
            x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            return x, y, z

    @property
    def camera_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the camera pose in the world coordinate system (i.e. the extrinsics matrix in the form of (translation, rotation)).

        Returns:
            A tuple of (translation, rotation). Rotation is represented by a matrix.
        """
        rotation = self.extrinsics[:3, :3]
        translation = self.extrinsics[:3, 3]
        return translation, rotation


@dataclass
class ObjectDetectionResult(object):
    """The result of object detection."""

    label: str
    """The label of the object."""

    pcd: o3d.geometry.PointCloud
    """The point cloud of the object."""

    reconstructed_pcd: Optional[o3d.geometry.PointCloud] = None
    """The reconstructed point cloud of the object."""

    reconstructed_mesh: Optional[o3d.geometry.TriangleMesh] = None
    """The mesh of the object."""

    pcd_mask: Optional[np.ndarray] = None

    @property
    def mesh(self) -> o3d.geometry.TriangleMesh:
        """Returns the mesh of the object. If the mesh is not reconstructed, an error will be raised."""

        if self.reconstructed_mesh is None:
            raise ValueError('The mesh is not reconstructed yet. Please call construct_object_meshes() first.')
        return self.reconstructed_mesh

    @property
    def mesh_bbox(self) -> o3d.geometry.AxisAlignedBoundingBox:
        """Returns the bounding box of the mesh of the object. If the mesh is not reconstructed, an error will be raised."""

        if self.reconstructed_mesh is None:
            raise ValueError('The mesh is not reconstructed yet. Please call construct_object_meshes() first.')
        return self.reconstructed_mesh.get_axis_aligned_bounding_box()


class ObjectCentricVisionPipeline(object):
    """The pipeline for object-centric vision."""

    def __init__(self, color_image: np.ndarray, depth_image: np.ndarray):
        """Initializes the pipeline.

        Args:
            color_image: The color image.
            depth_image: The depth image.
        """
        self.color_image = color_image
        self.depth_image = depth_image
        self.camera_transform = None
        self.foreground_ranges = None
        self._pcd = None
        self._foreground_segmentation = None
        self._table_model = None
        self._table_segmentation = None
        self._image_detected_objects = None
        self._detected_objects = None

    @property
    def pcd(self) -> o3d.geometry.PointCloud:
        """The point cloud of the scene."""
        if self._pcd is None:
            raise ValueError('Point cloud is not constructed yet. Please call construct_pointcloud() first.')
        return self._pcd

    @property
    def foreground_segmentation(self) -> np.ndarray:
        """The segmentation of the foreground, in the form of a boolean array of the same length as the point cloud."""
        if self._foreground_segmentation is None:
            raise ValueError('Background segmentation is not constructed yet. Please call construct_foreground_by_cropping() first.')
        return self._foreground_segmentation

    @cached_property
    def foreground_pcd(self) -> o3d.geometry.PointCloud:
        """The point cloud of the foreground."""
        return self.pcd.select_by_index(np.where(self.foreground_segmentation)[0])

    @property
    def table_model(self) -> np.ndarray:
        """The model of the table, in the form of a length-4 array representing the plane equation ax + by + cz + d = 0."""
        if self._table_model is None:
            raise ValueError('Table model is not constructed yet. Please call construct_table_segmentation() first.')
        return self._table_model

    @property
    def table_segmentation(self) -> np.ndarray:
        """The segmentation of the table, in the form of a boolean array of the same length as the point cloud."""
        if self._table_segmentation is None:
            raise ValueError('Table segmentation is not constructed yet. Please call construct_table_segmentation() first.')
        return self._table_segmentation

    @cached_property
    def tabletop_segmentation(self) -> np.ndarray:
        """The segmentation of the tabletop, in the form of a boolean array of the same length as the point cloud. Tabletop is defined as the foreground points that are not table points."""
        return ~self.table_segmentation & self.foreground_segmentation

    @cached_property
    def table_pcd(self) -> o3d.geometry.PointCloud:
        """The point cloud of the table."""
        return self.pcd.select_by_index(np.where(self.table_segmentation)[0])

    @cached_property
    def tabletop_pcd(self) -> o3d.geometry.PointCloud:
        """The point cloud of the tabletop. Tabletop is defined as the foreground points that are not table points."""
        return self.pcd.select_by_index(np.where(self.tabletop_segmentation)[0])

    @property
    def image_detected_objects(self) -> InstanceSegmentationResult:
        """The result of object detection in the image."""
        if self._image_detected_objects is None:
            raise ValueError('Object detection is not performed yet. Please call construct_object_segmentation() first.')
        return self._image_detected_objects

    @property
    def detected_objects(self) -> List[ObjectDetectionResult]:
        """The result of object detection in the scene."""
        if self._detected_objects is None:
            raise ValueError('Object detection is not performed yet. Please call construct_object_segmentation() first.')
        return self._detected_objects

    def construct_pointcloud(self, camera_transform: Optional[CameraTransformation] = None) -> o3d.geometry.PointCloud:
        """Constructs the point cloud from the color and depth images.

        Args:
            camera_transform: the camera transformation. If None, reconstruct the point cloud in the camera coordinate system.

        Returns:
            the point cloud.
        """
        self.camera_transform = camera_transform
        self._pcd = get_pointcloud(self.color_image, self.depth_image, camera_transform)
        return self._pcd

    def construct_foreground_by_cropping(
        self,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        z_range: Optional[Tuple[float, float]] = None,
        depth_range: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """Constructs the foreground segmentation by cropping the point cloud.

        Args:
            x_range: the range of x-axis in the world coordinate system.
            y_range: the range of y-axis in the world coordinate system.
            z_range: the range of z-axis in the world coordinate system.
            depth_range: the range of depth in the camera coordinate system.

        Returns:
            the foreground segmentation as a boolean array of shape (N,). N is the number of points in the point cloud.
        """
        depth_mask = np.ones(len(self.pcd.points), dtype=np.bool_)
        if depth_range is not None:
            depth_mask = depth_mask & (self.depth_image.flatten() > depth_range[0] * 1000) & (self.depth_image.flatten() < depth_range[1] * 1000)

        self.foreground_ranges = (x_range, y_range, z_range)
        self._foreground_segmentation = filter_pointcloud_range(self.pcd, x_range, y_range, z_range, return_condition=True) & depth_mask

        return self._foreground_segmentation

    def construct_table_segmentation(self, method: str = 'ransac', **kwargs):
        """Constructs the table segmentation.

        Args:
            method: the method to use for table segmentation. Available methods are {'ransac', 'threshold'}.

        Returns:
            the table segmentation as a boolean array of shape (N,). N is the number of points in the point cloud.
        """
        pcd = self.foreground_pcd
        if method == 'ransac':
            plane_model, raw_table_segmentation = ransac_table_detection(pcd, **kwargs, return_condition=True)
        elif method == 'threshold':
            plane_model, raw_table_segmentation = threshold_table_detection(pcd, **kwargs, return_condition=True)
        else:
            raise ValueError(f'Unknown method {method}. Available methods are {{ransac, threshold}}.')

        table_segmentation = np.zeros(len(self.pcd.points), dtype=np.bool_)
        table_segmentation[self.foreground_segmentation] = raw_table_segmentation
        self._table_model = plane_model
        self._table_segmentation = table_segmentation

        return self._table_segmentation

    def construct_object_segmentation(
        self, image_segmentation_model: Optional[ImageBasedPCDSegmentationModel] = None, detection_min_points: int = 50, enforce_detected_points: bool = True,
        dbscan: bool = True, dbscan_eps: float = 0.01, dbscan_min_samples: int = 500, dbscan_min_points: int = 1500, dbscan_with_sam: bool = True, sam_model: Optional[PointGuidedImageSegmentationModel] = None, sam_max_points: int = 100000,
        outlier_filter: bool = True, outlier_filter_radius: float = 0.005, outlier_filter_min_neighbors: int = 50,
        min_height: float = 0.015, verbose: bool = False
    ):
        """Constructs the object segmentation. The segmentation is performed in the following steps:

        1. If image_segmentation_model is not None, use the image segmentation model to segment the image.
        2. If dbscan is True, use DBSCAN to cluster the remaining points from the previous step.
        3. If outlier_filter is True, use radius outlier filter to remove outliers for each object obtained from the previous two steps.
        4. Use height filter to remove objects that are too thin.

        Args:
            image_segmentation_model: the image segmentation model. If None, skip the image segmentation step.
            detection_min_points: the minimum number of points for an object to be considered as a valid object.
            enforce_detected_points: if True, keep all points that are detected by the image segmentation model.
                Otherwise, only keep the points that are detected by the image segmentation model and are also in the "tabletop" segmentation, which equals foreground setminus table.
            dbscan: if True, use DBSCAN to cluster the remaining points from the previous step.
            dbscan_eps: the maximum distance between two samples for one to be considered as in the neighborhood of the other.
            dbscan_min_samples: the number of samples in a neighborhood for a point to be considered as a core point.
            dbscan_min_points: the minimum number of points for an object to be considered as a valid object.
            dbscan_with_sam: if True, use SAM to filter the points before DBSCAN.
            sam_model: the SAM model. If None, skip the SAM step.
            sam_max_points: the maximum number of points to use for SAM. This option is useful to avoid the table being detected as an object.
            outlier_filter: if True, use radius outlier filter to remove outliers for each object obtained from the previous two steps.
            outlier_filter_radius: the radius of the sphere that will determine which points are neighbors.
            outlier_filter_min_neighbors: the minimum number of neighbors that a point must have to be considered as an inlier.
            min_height: the minimum height of an object.
            verbose: if True, print out debug information.
        """
        remain_condition = self.tabletop_segmentation.copy()
        detected_objects = list()

        if image_segmentation_model is not None:
            results = image_segmentation_model.segment_image(self.color_image)
            self._image_detected_objects = results

            for i in range(results.nr_objects):
                label = results.pred_cls[i]

                if 'table' in label:
                    continue

                mask = results.masks[i].flatten()
                if enforce_detected_points:
                    mask = mask & self.foreground_segmentation
                else:
                    mask = mask & self.tabletop_segmentation

                if verbose:
                    print('Detected object with label', results.pred_cls[i], 'and mask size', mask.sum(), 'points.')

                if mask.sum() < detection_min_points or mask.sum() > sam_max_points:
                    continue
                pcd = self.pcd.select_by_index(np.where(mask)[0])
                detected_objects.append(ObjectDetectionResult(label, pcd, pcd_mask=mask))
                remain_condition = remain_condition & (~mask)

        # Run DBSCAN on the remaining points
        if dbscan:
            pcd = self.pcd.select_by_index(np.where(remain_condition)[0])
            if verbose:
                print('Running DBSCAN on remaining points with size', len(pcd.points))
            labels = np.asarray(pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_samples, print_progress=verbose))
            unique_labels = np.unique(labels)
            for label in unique_labels:
                if label == -1:
                    continue

                mask = labels == label
                label_name = f'unknown_{label}'
                global_mask = np.zeros(len(self.pcd.points), dtype=np.bool_)
                global_mask[remain_condition] = mask
                if global_mask.sum() < dbscan_min_points:
                    continue

                if dbscan_with_sam and sam_model is not None:
                    y, x = np.where(global_mask.reshape(self.color_image.shape[:2]))
                    y_mean, x_mean = int(y.mean()), int(x.mean())
                    if verbose:
                        print('  Running SAM on DBSCAN cluster with size', len(y), 'points.')
                    mask = sam_model.segment_from_point(self.color_image, (x_mean, y_mean))
                    if verbose:
                        print('  SAM mask size:', mask.sum())

                    if mask.sum() > sam_max_points:
                        if verbose:
                            print('  SAM mask too large, skipping.')
                        continue

                    mask = mask.flatten()
                    if enforce_detected_points:
                        mask = mask & self.foreground_segmentation
                    else:
                        mask = mask & self.tabletop_segmentation
                    global_mask = mask

                pcd = self.pcd.select_by_index(np.where(global_mask)[0])
                # pcd.paint_uniform_color(np.random.rand(3))
                detected_objects.append(ObjectDetectionResult(label_name, pcd, pcd_mask=global_mask))
                if verbose:
                    print('Detected object with label', label_name, 'and mask size', mask.sum(), 'points.')

        if verbose:
            print('Detected', len(detected_objects), 'objects before filtering.')

        # Remove outliers and thin objects
        filtered_objects = list()
        for obj in detected_objects:
            if outlier_filter:
                obj.pcd = remove_outliers(obj.pcd, radius=outlier_filter_radius, min_neighbors=outlier_filter_min_neighbors)
            if not dbscan_with_sam:
                min_bounds = obj.pcd.get_min_bound()
                max_bounds = obj.pcd.get_max_bound()

                if max_bounds[2] - min_bounds[2] < min_height:
                    if verbose:
                        print('Object', obj.label, 'is too thin, skipping.')
                    continue

            filtered_objects.append(obj)

        if verbose:
            print('Detected', len(filtered_objects), 'objects.')
        self._detected_objects = filtered_objects
        return self._detected_objects

    def construct_object_completion_by_extrusion(self, min_height: Optional[float] = 0.05, verbose: bool = False):
        """Construct object completion by extrusion the top surface of the table.

        Args:
            min_height: the minimum height of an object. If specified as None, skip the thin object filtering step.
                Otherwise, if an object is too thin, it will be extruded to the minimum height.
            verbose: if True, print out debug information.
        """
        for obj in self.detected_objects:
            if verbose:
                print('Shape reconstructing object', obj.label, 'with', len(obj.pcd.points), 'points.')
            obj.reconstructed_pcd = object_reconstruction_by_extrusion(obj.pcd, self.table_model, min_height=min_height)

    def construct_object_meshes(self, alpha: float = 0.1, verbose: bool = False):
        """Construct meshes for all detected objects using alpha shape.

        Args:
            alpha: the alpha value for alpha shape.
            verbose: if True, print out debug information.
        """
        for obj in self.detected_objects:
            if verbose:
                print('Mesh reconstruction for object', obj.label, 'with', len(obj.pcd.points), 'points.')
            obj.reconstructed_mesh = mesh_reconstruction_alpha_shape(obj.reconstructed_pcd, alpha=alpha)

    def export_scene(self, output_dir: str, verbose: bool = False):
        """Export the scene to a directory. All the metadata will be saved as "metainfo.json".

        Args:
            output_dir: the output directory.
        """
        io.mkdir(output_dir)
        metainfo = {
            'table_model': self.table_model.tolist(),
            'table_transformation': compute_transformation_from_plane_equation(*self.table_model).tolist(),
            'objects': list()
        }
        cv2.imwrite(osp.join(output_dir, 'color.png'), self.color_image[:, :, ::-1])
        cv2.imwrite(osp.join(output_dir, 'depth.png'), self.depth_image)
        builder = ObjectUrdfBuilder(output_dir)
        for idx, detection in jacinle.tqdm_gofor(self.detected_objects, desc='Exporting objects'):
            det_metainfo = {
                'index': idx,
                'label': detection.label,
                'unique_label': f'o{idx:02d}_{detection.label}',
                'pcd': f'{idx}_pcd.ply',
                'mesh': f'{idx}_mesh.ply',
                'mesh_obj': f'{idx}_mesh.obj',
                'urdf': f'{idx}_mesh.obj.urdf',
            }
            metainfo['objects'].append(det_metainfo)
            o3d.io.write_point_cloud(osp.join(output_dir, det_metainfo['pcd']), detection.pcd)
            mesh = detection.reconstructed_mesh
            mesh, center = canonize_mesh_center_(mesh)
            det_metainfo['pos'] = center.tolist()
            o3d.io.write_triangle_mesh(osp.join(output_dir, det_metainfo['mesh']), mesh)
            o3d.io.write_triangle_mesh(osp.join(output_dir, det_metainfo['mesh_obj']), mesh, write_triangle_uvs=True)

            with jacinle.cond_with(jacinle.suppress_stdout(), not verbose):  # if verbose, print out the output of urdf builder
                builder.build_urdf(osp.join(output_dir, det_metainfo['mesh_obj']), force_overwrite=True, decompose_concave=True, force_decompose=False, center=None)

        io.dump_json(osp.join(output_dir, 'metainfo.json'), metainfo)

    def visualize_pcd(self, name: str):
        """Visualize a point cloud. It will also visualize two coordinate frames: the camera frame and the world frame.

        Args:
            name: the name of the point cloud to visualize. Can be one of the following:
                - 'full': the full point cloud.
                - 'foreground': the foreground point cloud.
                - 'table': the table point cloud.
                - 'tabletop': the tabletop point cloud.
        """
        geometries = list()
        if name == 'full':
            geometries.append(self.pcd)
        elif name == 'foreground':
            geometries.append(self.foreground_pcd)
            # visualize the bounding box
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=(self.foreground_ranges[0][0], self.foreground_ranges[1][0], self.foreground_ranges[2][0]),
                max_bound=(self.foreground_ranges[0][1], self.foreground_ranges[1][1], self.foreground_ranges[2][1])
            )
            bbox.color = (0, 1, 0)
            geometries.append(bbox)
        elif name == 'table':
            pcd = self.table_pcd
            # pcd.paint_uniform_color([1, 0, 0])
            geometries.append(pcd)
        elif name == 'tabletop':
            pcd = self.tabletop_pcd
            geometries.append(pcd)
        else:
            raise ValueError(f'Unknown name {name}. Available names are {{full, foreground, table, tabletop}}.')

        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        geometries.append(world_frame)
        if self.camera_transform is not None:
            camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05).transform(np.linalg.inv(self.camera_transform.extrinsics))
            geometries.append(camera)

        # o3d.visualization.draw_geometries(geometries)
        visualize_with_camera_matrix(geometries, np.linalg.inv(self.camera_transform.extrinsics))

    def visualize_objects_raw_detection(self):
        """Visualize the raw detection results in 2D."""
        visualize_instance_segmentation(self.color_image, self.image_detected_objects)

    def visualize_objects_2d(self, rect_th: int = 3, text_size: int = 1, text_th: int = 3):
        """Visualize the detected objects back-projected to the color image.

        Args:
            rect_th: the thickness of the rectangle.
            text_size: the size of the text.
            text_th: the thickness of the text.
        """
        image = self.color_image.copy()
        for detection in self.detected_objects:
            label = detection.label
            masked_image = detection.pcd_mask.reshape(self.color_image.shape[:2])
            v, u = np.where(masked_image)
            rgb_mask = random_colored_mask(masked_image)
            image = cv2.addWeighted(image, 1, rgb_mask, 0.5, 0)
            cv2.rectangle(image, (u.min(), v.min()), (u.max(), v.max()), color=(0, 255, 0), thickness=rect_th)
            cv2.putText(image, label, (u.min(), v.min() + 10), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.tight_layout()
        plt.show()

    def visualize_objects_3d(self, name: str = 'mesh', overlay_background_pcd: bool = False):
        """Visualize the detected objects in 3D.

        Args:
            name: the name of the geometry to visualize. Can be one of the following:
                - 'mesh': the reconstructed mesh.
                - 'pcd': the reconstructed point cloud.
                - 'raw_pcd': the raw point cloud.
            overlay_background_pcd: whether to overlay the background point cloud.
        """
        print('Visualizing point cloud.')
        geometries = list()
        for obj in self.detected_objects:
            if name == 'mesh' and obj.reconstructed_mesh is not None:
                geometries.append(obj.reconstructed_mesh)
            elif name == 'pcd' and obj.reconstructed_pcd is not None:
                geometries.append(obj.reconstructed_pcd)
            elif name == 'raw_pcd':
                geometries.append(obj.pcd)

            # also add the bounding box
            bbox = geometries[-1].get_axis_aligned_bounding_box()
            bbox.color = (0, 1, 0)
            geometries.append(bbox)

        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        geometries.append(world_frame)
        if self.camera_transform is not None:
            camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05).transform(np.linalg.inv(self.camera_transform.extrinsics))
            geometries.append(camera)

        # visualize a table plane
        if self.table_model is not None:
            # table_model was estimated by RANSAC. So it's represented as a plane equation.
            # We need to convert it to a plane mesh.
            table_plane = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=0.001).translate(np.array([0, -0.5, 0]))
            table_plane.paint_uniform_color([0.5, 0.5, 0.5])
            T = compute_transformation_from_plane_equation(*self.table_model)
            table_plane = table_plane.transform(T)
            geometries.append(table_plane)

        if overlay_background_pcd:
            geometries.append(self.foreground_pcd)

        # o3d.visualization.draw_geometries(geometries)
        visualize_with_camera_matrix(geometries, np.linalg.inv(self.camera_transform.extrinsics))

    def run_default_pipeline(
        self, camera_transformation,
        x_range: Tuple[float, float] = (0.2, 1.0), y_range: Tuple[float, float] = (-0.5, 0.5), z_range: Tuple[float, float] = (-0.1, 0.5), depth_range: Tuple[float, float] = (0.05, 2.0),
        table_ransac_distance_threshold: float = 0.01, table_ransac_n: int = 3, table_ransac_num_iterations: int = 1000,
        image_segmentation_model: Optional[ImageBasedPCDSegmentationModel] = None, detection_min_points: int = 50, enforce_detected_points: bool = True,
        dbscan: bool = True, dbscan_eps: float = 0.01, dbscan_min_samples: int = 500, dbscan_min_points: int = 1500,
        dbscan_with_sam: bool = True, sam_model: Optional[PointGuidedImageSegmentationModel] = None, sam_max_points: int = 50000,
        outlier_filter: bool = True, outlier_filter_radius: float = 0.005, outlier_filter_min_neighbors: int = 50,
        min_height: float = 0.015,
        extrusion_min_height: float = 0.03,
        mesh_alpha: float = 0.1
    ):
        """Running the entire pipeline with the default parameters."""
        pbar = jacinle.tqdm_pbar(total=7, desc='Running object-centric vision pipeline', leave=False)
        with pbar:
            pbar.set_description('Reconstructing point cloud')
            self.construct_pointcloud(camera_transformation)
            pbar.update(); pbar.set_description('Segmenting the foreground')
            self.construct_foreground_by_cropping(x_range, y_range, z_range, depth_range)
            pbar.update(); pbar.set_description('Estimating the table plane')
            self.construct_table_segmentation(method='ransac', distance_threshold=table_ransac_distance_threshold, ransac_n=table_ransac_n, num_iterations=table_ransac_num_iterations)
            pbar.update(); pbar.set_description('Segmenting the objects')
            self.construct_object_segmentation(
                image_segmentation_model, detection_min_points=detection_min_points, enforce_detected_points=enforce_detected_points,
                dbscan=dbscan, dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples, dbscan_min_points=dbscan_min_points,
                dbscan_with_sam=dbscan_with_sam, sam_model=sam_model, sam_max_points=sam_max_points,
                outlier_filter=outlier_filter, outlier_filter_radius=outlier_filter_radius, outlier_filter_min_neighbors=outlier_filter_min_neighbors,
                min_height=min_height
            )
            pbar.update(); pbar.set_description('3D-reconstructing the objects')
            self.construct_object_completion_by_extrusion(min_height=extrusion_min_height)
            pbar.update(); pbar.set_description('Mesh-reconstructing the objects')
            self.construct_object_meshes(alpha=mesh_alpha)
            pbar.update(); pbar.set_description('Running NMS')
            self._detected_objects = mesh_nms(self.detected_objects, 0.1)


def get_pointcloud(color_image: np.ndarray, depth_image: np.ndarray, camera_transformation: Optional[CameraTransformation] = None) -> o3d.geometry.PointCloud:
    """Reconstruct a colored point cloud from a color and depth image.

    Args:
        color_image: a numpy array of shape (H, W, 3) representing the color image.
        depth_image: a numpy array of shape (H, W) representing the depth image.
        camera_transformation: a CameraTransformation object.

    Returns:
        a colored open3d point cloud.
    """

    xs = np.arange(0, color_image.shape[1])
    ys = np.arange(0, color_image.shape[0])
    xx, yy = np.meshgrid(xs, ys)
    xx = xx.flatten()
    yy = yy.flatten()
    zz = depth_image.flatten() / 1000.0
    colors = color_image[yy, xx, :].astype(np.float64) / 255.0

    if camera_transformation is not None:
        xx, yy, zz = camera_transformation.c2w(xx, yy, zz)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack([xx, yy, zz], axis=1))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def filter_pointcloud_range(
    pcd: o3d.geometry.PointCloud,
    x_range: Optional[Tuple[float, float]] = None, y_range: Optional[Tuple[float, float]] = None, z_range: Optional[Tuple[float, float]] = None,
    return_condition: bool = False
) -> Union[np.ndarray, o3d.geometry.PointCloud]:
    """Filter a point cloud by a range of x, y, z coordinates.

    Args:
        pcd: an open3d point cloud.
        x_range: a tuple of (min, max) x coordinates.
        y_range: a tuple of (min, max) y coordinates.
        z_range: a tuple of (min, max) z coordinates.
        return_condition: whether to return the condition used to filter the point cloud.

    Returns:
        an open3d point cloud or a numpy array of shape (N,) representing the condition.
    """

    condition = np.ones(len(pcd.points), dtype=np.bool_)
    xyz = np.asarray(pcd.points)
    if x_range is not None:
        condition &= (xyz[:, 0] >= x_range[0]) & (xyz[:, 0] <= x_range[1])
    if y_range is not None:
        condition &= (xyz[:, 1] >= y_range[0]) & (xyz[:, 1] <= y_range[1])
    if z_range is not None:
        condition &= (xyz[:, 2] >= z_range[0]) & (xyz[:, 2] <= z_range[1])

    if return_condition:
        return condition
    pcd = pcd.select_by_index(np.where(condition)[0])
    return pcd


def ransac_table_detection(
    pcd: o3d.geometry.PointCloud, distance_threshold: float = 0.005, ransac_n: int = 3, num_iterations: int = 1000, return_condition: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]]:

    """Detect the table plane using RANSAC.

    Args:
        pcd: an open3d point cloud.
        distance_threshold: the distance threshold for RANSAC.
        ransac_n: the number of points to sample for RANSAC.
        num_iterations: the number of iterations for RANSAC.
        return_condition: whether to return the condition used to filter the point cloud.

    Returns:
        If return_condition is True, returns a tuple of (model, condition). Condition is a numpy array of shape (N,) representing the condition.
        Otherwise, returns a tuple of (inliers, outliers). Inliers and outliers are open3d point clouds.
    """

    model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
    if return_condition:
        condition = np.zeros(len(pcd.points), dtype=np.bool_)
        condition[inliers] = True
        return model, condition

    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return inlier_cloud, outlier_cloud


def threshold_table_detection(
    pcd: o3d.geometry.PointCloud, z_threshold: float = 0.005, return_condition: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]]:
    """Detect the table plane using a simple threshold.

    Args:
        pcd: an open3d point cloud.
        z_threshold: the threshold for the z coordinate.
        return_condition: whether to return the condition used to filter the point cloud.

    Returns:
        If return_condition is True, returns a numpy array of shape (N,) representing the condition.
        Otherwise, returns a tuple of (inliers, outliers). Inliers and outliers are open3d point clouds.
    """

    xyz = np.asarray(pcd.points)
    condition = np.abs(xyz[:, 2]) < z_threshold

    if return_condition:
        # TODO(Jiayuan Mao): check whether this should be + or -.
        model = np.array([0, 0, 1, -z_threshold])
        return model, condition

    inlier_cloud = pcd.select_by_index(np.where(condition)[0])
    outlier_cloud = pcd.select_by_index(np.where(condition)[0], invert=True)

    return inlier_cloud, outlier_cloud


def project_pointcloud_to_plane(pcd: o3d.geometry.PointCloud, model: np.ndarray) -> o3d.geometry.PointCloud:
    """Project a point cloud to a plane (into a 2D shape)

    Args:
        pcd: an open3d point cloud.
        model: a numpy array of shape (4,) representing the plane model.

    Returns:
        an open3d point cloud.
    """

    xyz = np.asarray(pcd.points).copy()
    d = np.dot(xyz, model[:3]) + model[3]
    xyz -= d[:, None] * model[None, :3]
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(xyz)
    new_pcd.colors = pcd.colors
    return new_pcd


def _cdist(xyz1: np.ndarray, xyz2: np.ndarray) -> np.ndarray:
    """Compute the pairwise distance between two point clouds.

    Args:
        xyz1: a numpy array of shape (N, 3).
        xyz2: a numpy array of shape (M, 3).

    Returns:
        a numpy array of shape (N, M).
    """

    return np.sqrt(np.sum((xyz1[:, None, :] - xyz2[None, :, :]) ** 2, axis=-1))


def object_reconstruction_by_extrusion(
    pcd: o3d.geometry.PointCloud, model: np.ndarray, extrusion_distance: float = 0.005, min_height: Optional[float] = 0.05,
    include_inner_points: bool = False
) -> o3d.geometry.PointCloud:
    """Reconstruct an object by extrusion.

    Args:
        pcd: an open3d point cloud.
        model: a numpy array of shape (4,) representing the plane model.
        extrusion_distance: the distance to extrude.
        min_height: the minimum height of the extrusion. If an object is too thin, we will extrude it to this height.
        include_inner_points: whether to include the inner points of the object.
    """

    # Compute the max distance from the plane.
    xyz = np.asarray(pcd.points, dtype=np.float32).copy()
    d = np.abs(model[:3].dot(xyz.T) + model[3]) / np.linalg.norm(model[:3])
    max_d = np.max(d)

    # Extrude the point cloud.
    max_extrusion_nr = int(max_d / extrusion_distance)

    projection = project_pointcloud_to_plane(pcd, model)
    # Subsample the projected point cloud.
    projection = projection.voxel_down_sample(voxel_size=0.005)
    # projection = projection.uniform_down_sample(every_k_points=10)
    projected_xyz = np.asarray(projection.points)
    projected_rgb = np.asarray(projection.colors)

    points = [(projected_xyz, projected_rgb)]
    if include_inner_points:
        for i in range(max_extrusion_nr):
            extruded_xyz = projected_xyz + (i + 1) * extrusion_distance * model[:3]
            # Compute the distance of all points to the extruded point cloud.
            distances = _cdist(extruded_xyz, xyz)
            # Find the closest point on the original pcd for each point.
            closest_points = np.min(distances, axis=1)
            # If the closest point is too close, we stop extruding for this point.
            mask = closest_points > extrusion_distance
            extruded_xyz = extruded_xyz[mask]
            extruded_rgb = projected_rgb[mask]

            points.append((extruded_xyz, extruded_rgb))

            projected_xyz = projected_xyz[mask]
            projected_rgb = projected_rgb[mask]

    if min_height is not None and np.median(xyz[:, 2]) < min_height:
        # if the object is just too thin, we add a top surface.
        points.append((projected_xyz + np.array([0, 0, 0.05]), projected_rgb))
    else:
        points.append((xyz, np.asarray(pcd.colors)))

    # Concatenate all points.
    xyz = np.concatenate([p[0] for p in points], axis=0)
    rgb = np.concatenate([p[1] for p in points], axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points, pcd.colors = o3d.utility.Vector3dVector(xyz), o3d.utility.Vector3dVector(rgb)
    return pcd


def remove_outliers(pcd: o3d.geometry.PointCloud, min_neighbors: int, radius: float, verbose: bool = False) -> o3d.geometry.PointCloud:
    """Remove outliers from a point cloud.

    Note: this function is not used in the pipeline.

    Args:
        pcd: an open3d point cloud.
        min_neighbors: the minimum number of neighbors.
        radius: the radius to search for neighbors.

    Returns:
        an open3d point cloud.
    """

    if verbose:
        print("  Removing outliers... Before: ", len(pcd.points))
    pcd, _ = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    if verbose:
        print("  Removing outliers... After: ", len(pcd.points))
    return pcd


def mesh_reconstruction_alpha_shape(pcd: o3d.geometry.PointCloud, alpha: float = 0.1) -> o3d.geometry.TriangleMesh:
    """Reconstruct a mesh from a point cloud.

    Args:
        pcd: an open3d point cloud.
        alpha: the alpha value for the alpha shape.

    Returns:
        an open3d triangle mesh.
    """
    pcd = pcd.voxel_down_sample(voxel_size=0.0025)
    pcd.estimate_normals()
    # Project the object to the table plane
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=alpha)
    # t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    # t_filled = t_mesh.fill_holes(hole_size=1)
    # mesh = t_filled.to_legacy()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def mesh_nms(detected_objects: List[ObjectDetectionResult], bbox_overlapping_threshold: float = 0.1) -> List[ObjectDetectionResult]:
    """Perform non-maximum suppression on the detected objects.

    Args:
        detected_objects: a list of ObjectDetectionResult.
        bbox_overlapping_threshold: the threshold for overlapping.

    Returns:
        a list of ObjectDetectionResult.
    """

    # Sort the objects by their scores.
    # Compute the pairwise IoU.
    ious = np.zeros((len(detected_objects), len(detected_objects)))
    for i in range(len(detected_objects)):
        for j in range(i + 1, len(detected_objects)):
            ious[i, j] = ious[j, i] = compute_iou_3d(detected_objects[i].mesh_bbox, detected_objects[j].mesh_bbox)

    ious_visualization = [['' for _ in range(len(detected_objects) + 1)] for _ in range(len(detected_objects) + 1)]
    for i in range(len(detected_objects)):
        for j in range(len(detected_objects)):
            if i == j:
                ious_visualization[i + 1][j + 1] = 'X'
            else:
                ious_visualization[i + 1][j + 1] = f'{ious[i, j]:.2f}'
        ious_visualization[i + 1][0] = f'{detected_objects[i].label}'
        ious_visualization[0][i + 1] = f'{detected_objects[i].label}'
    print(jacinle.tabulate(ious_visualization, tablefmt='simple'))

    # Perform NMS.
    selected_objects_indices = []
    for i in range(len(detected_objects)):
        if len(selected_objects_indices) == 0:
            selected_objects_indices.append(i)
        else:
            if np.max(ious[i, selected_objects_indices]) > bbox_overlapping_threshold:
                continue
            selected_objects_indices.append(i)
    return [detected_objects[i] for i in selected_objects_indices]


def compute_iou_3d(bbox1: o3d.geometry.AxisAlignedBoundingBox, bbox2: o3d.geometry.AxisAlignedBoundingBox) -> float:
    """Compute the 3D IoU between two bounding boxes.

    Args:
        bbox1: an open3d axis aligned bounding box.
        bbox2: an open3d axis aligned bounding box.

    Returns:
        the 3D IoU.
    """

    # Compute the intersection.
    bbox1 = np.asarray(bbox1.get_box_points())
    bbox2 = np.asarray(bbox2.get_box_points())

    a, b = bbox1.min(axis=0), bbox1.max(axis=0)
    c, d = bbox2.min(axis=0), bbox2.max(axis=0)

    x, y = np.maximum(a, c), np.minimum(b, d)
    intersection = np.maximum(y - x, 0)
    intersection_volume = np.prod(intersection)

    a_volume = np.prod(b - a)
    b_volume = np.prod(d - c)

    union_volume = a_volume + b_volume - intersection_volume
    return intersection_volume / union_volume


def canonize_mesh_center_(mesh_: o3d.geometry.TriangleMesh) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """Canonize the mesh center. Note that this function modifies the mesh in place.

    Args:
        mesh_: an open3d triangle mesh.

    Returns:
        a tuple of an open3d triangle mesh, the center of the mesh.
    """

    mesh_copy = o3d.geometry.TriangleMesh(mesh_)
    # Compute the center of the mesh.
    center = mesh_copy.get_center()
    # Compute the transformation matrix.
    T = np.eye(4)
    T[:3, 3] = -center
    # Apply the transformation matrix.
    mesh_copy.transform(T)
    return mesh_copy, center


def compute_transformation_from_plane_equation(a, b, c, d):
    """Compute the transformation matrix from a plane equation.

    Args:
        a, b, c, d: the plane equation parameters.

    Returns:
        a 4x4 transformation matrix.
    """
    # Normal of the plane
    N = np.array([a, b, c])
    # Normalize the normal vector
    n = N / np.linalg.norm(N)
    # Z axis
    z_axis = np.array([0, 0, 1])
    # Axis around which to rotate - cross product between Z axis and n
    axis = np.cross(z_axis, n)
    axis = axis / np.linalg.norm(axis)  # normalize the axis
    # Angle between Z axis and n
    angle = np.arccos(np.dot(z_axis, n))
    # Compute rotation matrix using scipy's Rotation
    rotation_matrix = quat2mat(axisangle2quat(axis, angle))
    # Compute translation vector
    translation_vector = -d * n
    # Construct transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    return transformation_matrix


def visualize_with_camera_matrix(geometries, camera_extrinsics):
    """Visualize the geometries with a camera matrix.

    Args:
        geometries: a list of open3d geometries.
        camera_extrinsics: a 4x4 camera extrinsics matrix.
    """
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().background_color = np.asarray([0, 0, 0])

    view_ctl = vis.get_view_control()

    for geom in geometries:
        vis.add_geometry(geom)

    cam = view_ctl.convert_to_pinhole_camera_parameters()
    cam.extrinsic = camera_extrinsics
    view_ctl.convert_from_pinhole_camera_parameters(cam)

    vis.run()
    vis.destroy_window()


def load_scene_in_pybullet(client: BulletClient, scene_dir: str, verbose: bool = False, simplify_name: bool = True, static: bool = False) -> Dict[str, Dict]:
    scene_metafile = osp.join(scene_dir, 'metainfo.json')
    scene_metainfo = jacinle.load(scene_metafile)

    unique_labels = set()
    non_unique_labels = set()
    non_unique_label_counts = dict()

    if simplify_name:
        for o in scene_metainfo['objects']:
            label = o['label'].replace(' ', '_')
            if label in unique_labels:
                non_unique_labels.add(label)
                non_unique_label_counts[label] = 0
                unique_labels.remove(label)
            else:
                unique_labels.add(label)

    load_metainfo = dict()
    for o in scene_metainfo['objects']:
        urdf_filename = osp.basename(o['urdf'])
        urdf_filename = osp.join(scene_dir, urdf_filename)
        pos = o['pos']
        pos = (pos[0], pos[1], pos[2] + 0.01)

        if simplify_name:
            name = o['label'].replace(' ', '_')
            if name in non_unique_labels:
                non_unique_label_counts[name] += 1
                name = name + '_' + str(non_unique_label_counts[name])
            else:
                assert name in unique_labels
        else:
            name = o['unique_label'].replace(' ', '_')

        index = client.load_urdf(urdf_filename, body_name=name, pos=pos, static=static)
        load_metainfo[name] = dict(id=index, label=o['label'])

    if verbose:
        print('Loaded objects:')
        for name, info in load_metainfo.items():
            print(f'  {name}: {info["label"]}')

    return load_metainfo

