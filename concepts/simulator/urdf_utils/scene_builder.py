#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : scene_builder.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/03/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""A set of utility classes and functions for saving and loading vision-pipeline outputs into simulators."""

import os
import os.path as osp
import numpy as np
import open3d as o3d

from typing import Optional, Union
from dataclasses import dataclass

import jacinle
from concepts.utils.typing_utils import Vec3f, Vec4f, Open3DPointCloud
from concepts.simulator.urdf_utils.obj2urdf import ObjectUrdfBuilder


class SceneItem(object):
    pass


@dataclass
class RobotItem(SceneItem):
    identifier: str
    name: str
    pos: Vec3f
    quat_xyzw: Vec4f


@dataclass
class TableItem(SceneItem):
    identifier: str
    name: str
    size: Vec3f
    pos: Vec3f
    quat_xyzw: Vec4f


@dataclass
class ObjectItem(SceneItem):
    identifier: str
    name: str
    pos: Vec3f
    quat_xyzw: Vec4f
    tags: tuple[str, ...]

    pcd_ply_filename: Optional[str]
    mesh_ply_filename: str
    mesh_obj_filename: str
    urdf_filename: str


class SceneBuilder(object):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.object_urdf_builder = ObjectUrdfBuilder(output_dir)
        self.robots = dict()
        self.objects = dict()

    robots: dict[str, RobotItem]
    objects: dict[str, Union[TableItem, ObjectItem]]

    def add_robot(self, identifier: str, name: str, pos: Vec3f, quat_xyzw: Vec4f):
        if identifier in self.robots:
            raise ValueError(f'Robot with identifier {identifier} already exists.')
        self.robots[identifier] = RobotItem(identifier, name, _as_tuple(pos), _as_tuple(quat_xyzw))

    def add_table(self, identifier: str, name: str, size: Vec3f, surface_center_pos: Vec3f, surface_center_quat_xyzw: Vec4f):
        if identifier in self.objects:
            raise ValueError(f'Object with identifier {identifier} already exists.')
        pos = surface_center_pos - size / 2
        quat = surface_center_quat_xyzw
        self.objects[identifier] = TableItem(identifier, name, size, _as_tuple(pos), _as_tuple(quat))

    def add_mesh_open3d(
        self, name: str, mesh: o3d.geometry.TriangleMesh, pointcloud: Optional[Open3DPointCloud] = None,
        pos: Optional[Vec3f] = None, quat_xyzw: Optional[Vec4f] = None,
        identifier: Optional[str] = None, tags: tuple[str, ...] = tuple(),
        verbose: bool = False
    ):
        if identifier is None:
            identifier = self._gen_identifier(name)
        if identifier in self.objects:
            raise ValueError(f'Object with identifier {identifier} already exists.')

        pcd_ply_filename = osp.join(self.output_dir, f'{identifier}_pcd.ply')
        mesh_ply_filename = osp.join(self.output_dir, f'{identifier}.ply')
        mesh_obj_filename = osp.join(self.output_dir, f'{identifier}.obj')

        if pos is None:
            mesh, pos = canonize_mesh_center(mesh)
        if quat_xyzw is None:
            quat_xyzw = [0, 0, 0, 1]

        pos, quat_xyzw = _as_tuple(pos), _as_tuple(quat_xyzw)

        o3d.io.write_triangle_mesh(mesh_ply_filename, mesh)
        o3d.io.write_triangle_mesh(mesh_obj_filename, mesh)
        if pointcloud is not None:
            o3d.io.write_point_cloud(pcd_ply_filename, pointcloud)

        with jacinle.cond_with(jacinle.suppress_stdout(), not verbose):  # if verbose, print out the output of urdf builder
            self.object_urdf_builder.build_urdf(
                mesh_obj_filename,
                force_overwrite=True, decompose_concave=True, force_decompose=False, center=None
            )
            urdf_filename = mesh_obj_filename + '.urdf'

        self.objects[identifier] = ObjectItem(identifier, name, pos, quat_xyzw, tags, pcd_ply_filename, mesh_ply_filename, mesh_obj_filename, urdf_filename)

    def add_pointcloud_open3d(
        self, name: str, pointcloud: Open3DPointCloud,
        pos: Optional[Vec3f] = None, quat_xyzw: Optional[Vec4f] = None,
        identifier: Optional[str] = None, tags: tuple[str, ...] = tuple()
    ):
        mesh = mesh_reconstruction_alpha_shape(pointcloud)
        self.add_mesh_open3d(name, mesh, pointcloud, pos=pos, quat_xyzw=quat_xyzw, identifier=identifier, tags=tags)

    def _gen_identifier(self, name: str) -> str:
        if name not in self.objects:
            return name
        i = 1
        while f'{name}_{i}' in self.objects:
            i += 1
        return f'{name}_{i}'

    def export_metadata(self, filename: Optional[str] = None):
        if filename is None:
            filename = osp.join(self.output_dir, 'metadata.json')
        metadat = dict()
        metadat['robots'] = [vars(robot) for robot in self.robots.values()]
        metadat['objects'] = [vars(obj) for obj in self.objects.values()]
        jacinle.io.dump_json(filename, metadat)

    @staticmethod
    def load_metadata(dirname_or_filename: str):
        if osp.isfile(dirname_or_filename):
            dirname = osp.dirname(dirname_or_filename)
            filename = dirname_or_filename
        else:
            dirname = dirname_or_filename
            filename = osp.join(dirname_or_filename, 'metadata.json')
        metadat = jacinle.io.load_json(filename)
        builder = SceneBuilder(osp.dirname(filename))
        for robot in metadat['robots']:
            builder.robots[robot['identifier']] = RobotItem(**robot)
        for obj in metadat['objects']:
            builder.objects[obj['identifier']] = ObjectItem(**obj)
        return builder


def canonize_mesh_center(mesh_: o3d.geometry.TriangleMesh) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
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


def _as_tuple(v: Union[np.ndarray, tuple, list]) -> tuple:
    if isinstance(v, (tuple, list)):
        return tuple(v)
    if isinstance(v, np.ndarray):
        return tuple(v.tolist())
    raise ValueError(f'Invalid type: {type(v)}')
