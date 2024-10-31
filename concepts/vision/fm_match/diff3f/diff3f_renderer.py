#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : diff3f_renderer.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/16/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Renderer 3D meshes with the PyTorch3D renderer."""

import math
from typing import Optional, Union

import torch
import torch.nn as nn

from pytorch3d.structures.meshes import Meshes
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes

from pytorch3d.renderer.blending import BlendParams
from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.utils import TensorProperties
from pytorch3d.renderer.mesh.rasterizer import Fragments, RasterizationSettings, MeshRasterizer
from pytorch3d.renderer.mesh.shader import HardPhongShader
from pytorch3d.renderer.mesh.renderer import MeshRenderer


from concepts.vision.fm_match.diff3f.diff3f_mesh import MeshContainer


class HardPhongNormalShader(nn.Module):
    """
    Modifies HardPhongShader to return normals

    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardPhongShader(device=torch.device("cuda:0"))
    """

    def __init__(
        self,
        device = "cpu",
        cameras: Optional[TensorProperties] = None,
        lights: Optional[TensorProperties] = None,
        materials: Optional[Materials] = None,
        blend_params: Optional[BlendParams] = None,
    ) -> None:
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        cameras = self.cameras
        if cameras is not None:
            self.cameras = cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self

    def phong_normal_shading(self, meshes, fragments) -> torch.Tensor:
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]
        ones = torch.ones_like(fragments.bary_coords)
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, ones, faces_normals
        )
        return pixel_normals

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardPhongShader"
            raise ValueError(msg)
        normals = self.phong_normal_shading(
            meshes=meshes,
            fragments=fragments,
        )
        return normals


@torch.no_grad()
def run_rendering(device: str, mesh: Meshes, num_views: int, H: int, W: int, additional_angle_azi: float = 0, additional_angle_ele: float = 0, use_normal_map: bool = False):
    bbox = mesh.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bb_diff = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2.0
    scaling_factor = 0.65
    distance = torch.sqrt((bb_diff * bb_diff).sum())
    distance *= scaling_factor
    steps = int(math.sqrt(num_views))
    end = 360 - 360/steps
    elevation = torch.linspace(start = 0 , end = end , steps = steps).repeat(steps) + additional_angle_ele
    azimuth = torch.linspace(start = 0 , end = end , steps = steps)
    azimuth = torch.repeat_interleave(azimuth, steps) + additional_angle_azi
    bbox_center = bbox_center.unsqueeze(0)
    rotation, translation = look_at_view_transform(
        dist=distance, azim=azimuth, elev=elevation, device=device, at=bbox_center
    )
    camera = PerspectiveCameras(R=rotation, T=translation, device=device)
    rasterization_settings = RasterizationSettings(
        image_size=(H, W), blur_radius=0.0, faces_per_pixel=1, bin_size=0
    )
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=rasterization_settings)
    camera_centre = camera.get_camera_center()
    lights = PointLights(
        diffuse_color=((0.4, 0.4, 0.5),),
        ambient_color=((0.6, 0.6, 0.6),),
        specular_color=((0.01, 0.01, 0.01),),
        location=camera_centre,
        device=device,
    )
    shader = HardPhongShader(device=device, cameras=camera, lights=lights)
    batch_renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    batch_mesh = mesh.extend(num_views)
    normal_batched_renderings = None
    batched_renderings = batch_renderer(batch_mesh)
    if use_normal_map:
        normal_shader = HardPhongNormalShader(device=device, cameras=camera, lights=lights)
        normal_batch_renderer = MeshRenderer(rasterizer=rasterizer, shader=normal_shader)
        normal_batched_renderings = normal_batch_renderer(batch_mesh)
    fragments = rasterizer(batch_mesh)
    depth = fragments.zbuf
    return batched_renderings, normal_batched_renderings, camera, depth


def batch_render(device: str, mesh: Union[MeshContainer, Meshes], num_views: int, H: int, W: int, use_normal_map: bool = False):
    if isinstance(mesh, MeshContainer):
        mesh = mesh.to_pytorch3d_meshes(device)

    trials = 0
    additional_angle_azi = 0.
    additional_angle_ele = 0.
    while trials < 5:
        try:
            return run_rendering(
                device, mesh, num_views, H, W,
                additional_angle_azi=additional_angle_azi, additional_angle_ele=additional_angle_ele,
                use_normal_map=use_normal_map
            )
        except torch.linalg.LinAlgError as e:
            trials += 1
            print("lin alg exception at rendering, retrying ", trials)
            additional_angle_azi = torch.randn(1).item()
            additional_angle_ele = torch.randn(1).item()
            continue

