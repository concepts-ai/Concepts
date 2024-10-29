#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : mesh_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/01/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import tempfile
import trimesh
import open3d as o3d
import numpy as np

from typing import Optional, Union, Tuple
from concepts.math.rotationlib_xyzw import axisangle2quat, quat2mat
from concepts.utils.typing_utils import Open3DPointCloud, Open3DTriangleMesh, Trimesh, Vec3f


def mesh_line_intersect(t_mesh: o3d.t.geometry.TriangleMesh, ray_origin: np.ndarray, ray_direction: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Intersects a ray with a mesh.

    Args:
        t_mesh: the mesh to intersect with.
        ray_origin: the origin of the ray.
        ray_direction: the direction of the ray.

    Returns:
        A tuple of (point, normal) if an intersection is found, None otherwise.
    """

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)

    ray = o3d.core.Tensor.from_numpy(np.array(
        [[ray_origin[0], ray_origin[1], ray_origin[2], ray_direction[0], ray_direction[1], ray_direction[2]]],
        dtype=np.float32
    ))
    result = scene.cast_rays(ray)

    # no intersection.
    if result['geometry_ids'][0] == scene.INVALID_ID:
        return None

    inter_point = np.asarray(ray_origin) + np.asarray(ray_direction) * result['t_hit'][0].item()
    inter_normal = result['primitive_normals'][0].numpy()
    return inter_point, inter_normal


def np2open3d_pcd(points: np.ndarray, colors: Optional[np.ndarray] = None) -> Open3DPointCloud:
    """Generate an Open3D point cloud from numpy arrays.

    Args:
        points: the points to add to the point cloud.
        colors: the colors to add to the point cloud.

    Returns:
        An Open3D point cloud.
    """
    pcd = o3d.geometry.PointCloud
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def trimesh_to_open3d_mesh(trimesh_mesh: Trimesh) -> Open3DTriangleMesh:
    """Convert a Trimesh mesh to an Open3D mesh.

    Args:
        trimesh_mesh: the Trimesh mesh to convert.

    Returns:
        An Open3D mesh.
    """

    # Check if the input is a Trimesh mesh
    if not isinstance(trimesh_mesh, trimesh.Trimesh):
        raise TypeError("Input must be a Trimesh mesh")

    open3d_mesh = o3d.geometry.TriangleMesh()
    open3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    open3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)

    # Check and transfer vertex normals, if they exist
    if trimesh_mesh.vertex_normals.size > 0:
        open3d_mesh.vertex_normals = o3d.utility.Vector3dVector(trimesh_mesh.vertex_normals)

    # Check and transfer vertex colors, if they exist
    # Trimesh stores colors in the 'visual' attribute
    if hasattr(trimesh_mesh.visual, 'vertex_colors') and trimesh_mesh.visual.vertex_colors.size > 0:
        # Open3D expects colors in the range [0, 1], but Trimesh uses [0, 255]
        vertex_colors_normalized = trimesh_mesh.visual.vertex_colors[:, :3] / 255.0
        open3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors_normalized)

    return open3d_mesh


def open3d_mesh_to_trimesh(open3d_mesh: Open3DTriangleMesh) -> Trimesh:
    """Convert an Open3D mesh to a Trimesh mesh.

    Args:
        open3d_mesh: the Open3D mesh to convert.

    Returns:
        A Trimesh mesh.
    """

    # Check if the input is an Open3D mesh
    if not isinstance(open3d_mesh, o3d.geometry.TriangleMesh):
        raise TypeError("Input must be an Open3D TriangleMesh")

    # Convert Open3D mesh to Trimesh by extracting vertices and faces
    vertices = np.asarray(open3d_mesh.vertices)
    faces = np.asarray(open3d_mesh.triangles)

    # Create a Trimesh object using the extracted vertices and faces
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    return trimesh_mesh


def set_open3d_mesh_camera(
    vis: o3d.visualization.Visualizer, look_at: Vec3f = (0, 0, 0), distance: float = 1, fov: float = 60, elevation: float = 0., azimuth: float = 0., yaw: float = 0.
):
    """Set the camera of an Open3D mesh.

    Args:
        vis: the Open3D visualizer to set the camera for.
        look_at: the point to look at.
        distance: the distance of the camera from the mesh.
        fov: the field of view of the camera.
        elevation: the elevation of the camera.
        azimuth: the azimuth of the camera.
        yaw: the yaw of the camera.
    """

    ctr = vis.get_view_control()
    ctr.change_field_of_view(fov - ctr.get_field_of_view())
    ctr.set_lookat(look_at)

    # Compute the up and front based on the elevation and azimuth
    front = np.array([
        np.cos(np.radians(azimuth)) * np.cos(np.radians(elevation)),
        np.sin(np.radians(azimuth)) * np.cos(np.radians(elevation)),
        np.sin(np.radians(elevation))
    ])

    up = np.array([0, 0, 1])
    if np.abs(np.dot(front, up)) > 1e-6:
        up = np.array([0, 1, 0])
    up = np.cross(front, np.cross(up, front))

    assert np.dot(front, up) < 1e-6, f"Front and up vectors are not orthogonal: {front=} {up=} {np.dot(front, up)}"

    # Rotate the up vector by the yaw angle around the front vector
    rotate_quat = axisangle2quat(front, np.radians(yaw))
    rotate_mat = quat2mat(rotate_quat)
    up = np.dot(rotate_mat, up)

    ctr.set_front(front)
    ctr.set_up(up)

    ctr.set_zoom(distance)


def render_open3d_mesh(
    obj: Union[Open3DTriangleMesh, Open3DPointCloud], width: int = 512, height: int = 512,
    look_at: Vec3f = (0, 0, 0), distance: float = 1, fov: float = 60,
    elevation: float = 0., azimuth: float = 0., yaw: float = 0.,
) -> np.ndarray:
    assert isinstance(obj, (o3d.geometry.TriangleMesh, o3d.geometry.PointCloud))

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(obj)
    set_open3d_mesh_camera(vis, look_at, distance, fov, elevation, azimuth, yaw)
    vis.poll_events()
    vis.update_renderer()

    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        vis.capture_screen_image(f.name)
        vis.destroy_window()

        image = o3d.io.read_image(f.name)
    return np.asarray(image).copy()
