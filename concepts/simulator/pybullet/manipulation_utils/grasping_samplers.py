#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : grasping_samplers.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/06/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
import trimesh
import open3d as o3d

from dataclasses import dataclass
from typing import Union, Optional, Tuple

from concepts.simulator.pybullet.rotation_utils import get_quaternion_from_matrix, find_orthogonal_vector
from concepts.simulator.pybullet.components.robot_base import Robot


@dataclass
class GraspGeneratorConfig(object):
    pass


@dataclass
class GraspReturn(object):
    point1: np.ndarray
    normal1: np.ndarray
    point2: np.ndarray
    normal2: np.ndarray
    ee_pos: np.ndarray
    ee_quat: np.ndarray
    qpos: np.ndarray


class GraspSampler(object):
    """Generate grasps for the input mesh."""

    def __init__(self, robot: Robot, gripper_size: float, gripper_gap: float, reachability_min_bound: Optional[np.ndarray] = None, reachability_max_bound: Optional[np.ndarray] = None):
        """Initialize the grasp generator.

        Args:
            robot: the robot instance.
            gripper_size: the size of the gripper (assumed to be a square contact surface).
            gripper_gap: the gap between the gripper fingers.
            reachability_min_bound: the minimum bound of the reachable workspace. If specified, will be used to filter out grasps that are not reachable.
            reachability_max_bound: the maximum bound of the reachable workspace. If specified, will be used to filter out grasps that are not reachable.
        """
        self.robot = robot
        self.gripper_size = gripper_size
        self.gripper_gap = gripper_gap

        self.reachability_min_bound = reachability_min_bound
        self.reachability_max_bound = reachability_max_bound

    def sample_grasp(self, input_shape: Union[np.ndarray, o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]) -> Optional[GraspReturn]:
        """Sample a grasp from the input point cloud, or mesh

        Args:
            input_pc: the input point cloud.

        Returns:
            GraspReturn: The sampled grasp.
        """

        if isinstance(input_shape, np.ndarray):
            input_shape = o3d.geometry.PointCloud()
            input_shape.points = o3d.utility.Vector3dVector(input_shape)
            input_shape.estimate_normals()

        if self.reachability_min_bound is not None and self.reachability_max_bound is not None:
            if isinstance(input_shape, o3d.geometry.TriangleMesh):
                aabb = o3d.geometry.AxisAlignedBoundingBox(self.reachability_min_bound, self.reachability_max_bound)
                input_shape = input_shape.crop(aabb)
                if len(input_shape.triangles) < 10:
                    return None
            elif isinstance(input_shape, o3d.geometry.PointCloud):
                aabb = o3d.geometry.AxisAlignedBoundingBox(self.reachability_min_bound, self.reachability_max_bound)
                input_shape = input_shape.crop(aabb)
                if len(input_shape.points) < 10:
                    return None
            else:
                raise TypeError(f'Unknown supported type {type(input_shape)}.')

        return self._sample_grasp_pcd(input_shape)

    def _sample_grasp_pcd(self, input_pc: o3d.geometry.PointCloud) -> Optional[GraspReturn]:
        raise TypeError('PCD grasp sampling is not implemented.')

    def _sample_grasp_mesh(self, input_mesh: o3d.geometry.TriangleMesh) -> Optional[GraspReturn]:
        raise TypeError('Mesh grasp sampling is not implemented.')


@dataclass
class LocalConvexificationGraspGeneratorConfig(GraspGeneratorConfig):
    """The minimum number of points in the point cloud to perform local convexification."""
    min_pointcloud_size: int = 10

    """The minimum gap between two gripper attachment points."""
    min_gripper_gap: float = 0.01

    """The minimum difference between the two normals of the gripper attachment points."""
    min_normal_difference: float = 0.9

    """The minimum quaternion norm."""
    min_quat_norm: float = 0.5

    """The maximum difference between the ik solution and the target."""
    max_ik_diff: float = 0.02

    """If true, the z-axis of the gripper should point to the front."""
    constraint_pos_x: bool = True

    """Visualize the configuration returned by the grasp generator."""
    debug_vis_return: bool = False

    """Visualize the each configuration tested by the grasp generator."""
    debug_vis_trials: bool = False


class LocalConvexificationGraspSampler(GraspSampler):
    config: LocalConvexificationGraspGeneratorConfig

    def __init__(
        self, robot: Robot,
        gripper_size: float, gripper_gap: float,
        reachability_min_bound: Optional[np.ndarray] = None, reachability_max_bound: Optional[np.ndarray] = None,
        config: Optional[LocalConvexificationGraspGeneratorConfig] = None,
        **kwargs
    ):
        super().__init__(robot, gripper_size, gripper_gap, reachability_min_bound, reachability_max_bound)

        if config is None:
            self.config = LocalConvexificationGraspGeneratorConfig(**kwargs)
        else:
            assert len(kwargs) == 0, 'Cannot specify both config and kwargs.'
            self.config = config

    def _sample_grasp_pcd(self, pcd: o3d.geometry.PointCloud, trials: int = 100) -> Optional[GraspReturn]:
        for _ in range(trials):
            # Sample a point in the point cloud.
            index = np.random.randint(0, len(pcd.points))
            point1 = pcd.points[index]
            normal1 = -pcd.normals[index]

            # Make a local crop of the point cloud.
            min_bound = point1 - self.gripper_size * 2
            max_bound = point1 + self.gripper_size * 2
            pcd_crop = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))

            if len(pcd_crop.points) < self.config.min_pointcloud_size:
                continue

            # Perform local convexification.
            # print(np.asarray(pcd_crop.points))
            # print(pcd_crop.get_axis_aligned_bounding_box())
            hull, _ = pcd_crop.compute_convex_hull()

            # Create the triangular mesh with the vertices and faces from open3d.
            tmesh = trimesh.Trimesh(np.asarray(hull.vertices), np.asarray(hull.triangles), vertex_normals=np.asarray(hull.vertex_normals))
            # Ray casting the point cloud to find the other point on the mesh.
            intersector = trimesh.ray.ray_triangle.RayMeshIntersector(tmesh)
            index, point2 = _trimesh_ray_casting(intersector, point1, normal1)

            if index is not None:
                normal2 = tmesh.face_normals[index]

                # Filter the gap.
                gap = np.linalg.norm(point1 - point2)
                if gap < self.config.min_gripper_gap:
                    continue
                if gap > self.gripper_gap:
                    continue

                # Filter the normal.
                if np.abs(np.dot(normal1, normal2)) < self.config.min_normal_difference:
                    continue

                # Construct three axes for the gripper.
                ee_pos = (point1 + point2) / 2
                ee_d = (point2 - point1) / np.linalg.norm(point2 - point1)
                ee_u = find_orthogonal_vector(point2 - point1)
                ee_v = np.cross(ee_d, ee_u)

                with self.robot.client.w.save_world() as world_saver:
                    for ee_norm1 in [ee_u, ee_v, -ee_u, -ee_v]:
                        if self.config.debug_vis_trials:
                            world_saver.restore()

                        ee_norm2 = np.cross(ee_d, ee_norm1)

                        # if self.config.constraint_pos_x and ee_norm2[0] < 0:
                        #     continue
                        # if self.config.constraint_pos_x and ee_norm2[0] > 0:
                        #     ee_norm2 = -ee_norm2
                        #     ee_norm1 = -ee_norm1

                        ee_quat = self._construct_quat(ee_norm2, ee_d, ee_norm1)

                        if self.config.debug_vis_trials:
                            print(f'Testing x={ee_norm2}, y={ee_d}, z={ee_norm1}, point1={point1}, point2={point2}.')

                        ee_grasp = self._check_collision(ee_pos, ee_quat)

                        if self.config.debug_vis_trials:
                            self.robot.client.step(2)
                            input('Press enter to continue.')

                        if ee_grasp is not None:
                            return GraspReturn(point1, normal1, point2, normal2, *ee_grasp)
            else:
                continue
        return None

    def _construct_quat(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Construct a quaternion from the new x-y-z coordinate axes of the gripper."""
        return get_quaternion_from_matrix(np.stack([x, y, z], axis=1))

    def _check_collision(self, ee_pos, ee_quat) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Check if the grasping is collision-free."""

        if np.linalg.norm(ee_quat) < self.config.min_quat_norm:
            if self.config.debug_vis_trials:
                print('Quaternion checking failed.')
            return None

        qpos = self.robot.ik(ee_pos, ee_quat, force=True)

        if qpos is None:
            if self.config.debug_vis_trials:
                print('IK failed.')
            return None

        self.robot.set_qpos(qpos)
        if np.linalg.norm(self.robot.get_ee_pose()[0] - ee_pos) > self.config.max_ik_diff:
            if self.config.debug_vis_trials:
                print('IK post checking failed: ', np.linalg.norm(self.robot.get_ee_pose()[0] - ee_pos))
            return None

        self.robot.client.p.performCollisionDetection()

        # self.robot.client.step(1)
        # input('Press enter to continue')

        contacts = self.robot.client.w.get_contact(a=self.robot.get_robot_body_id())

        all_contact_bodies = {c.body_b for c in contacts} - {self.robot.get_robot_body_id()}
        all_contact_body_names = [self.robot.client.w.body_names[b] for b in all_contact_bodies]

        if self.config.debug_vis_return:
            if len(all_contact_bodies) == 0:
                print('No contact')
                print('-' * 80)
                print(*contacts, sep='\n')
                print(all_contact_body_names)
                self.robot.client.step(5)
        if self.config.debug_vis_trials:
            print('Contacts')
            for c in contacts:
                print(c.body_a_name, c.body_b_name, c.link_a_name, c.link_b_name, c.position_on_a, c.position_on_b)

        if len(all_contact_body_names) > 0:
            return None

        return ee_pos, ee_quat, qpos


def _trimesh_ray_casting(intersector, point, normal):
    """Ray casting using trimesh ray interesector."""

    index, _, locations = intersector.intersects_id([point], [normal], return_locations=True, multiple_hits=True)
    if len(index) > 0:
        if len(index) > 1:
            distances = np.linalg.norm(locations - point, axis=1)
            i = np.argmax(distances)
            return index[i], locations[i]
        return index[0], locations[0]
    return None, None

