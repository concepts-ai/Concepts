#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : diff3f_extractor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/16/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import random
import numpy as np
import time
from tqdm import tqdm
from typing import Optional, Union, Tuple, List
from PIL import Image

from pytorch3d.structures import Meshes
from pytorch3d.ops import ball_query

# from diffusion import add_texture_to_render
from concepts.vision.fm_match.diff3f.diff3f_mesh import MeshContainer
from concepts.vision.fm_match.diff3f.extractor_dino import get_dino_features
from concepts.vision.fm_match.diff3f.diff3f_renderer import batch_render
from concepts.vision.fm_match.diff3f.diff3f_utils import get_maximal_distance

FEATURE_DIFFUSION_DIMS = 1280
FEATURE_DINO_DIMS = 768
FEATURE_DIMS = FEATURE_DIFFUSION_DIMS + FEATURE_DINO_DIMS


def arange_pixels(
    resolution: Tuple[int, int] = (128, 128),
    batch_size: int = 1,
    subsample_to: Optional[int] = None,
    invert_y_axis: bool = False,
    margin: float = 0,
    align_corners: bool = True,
    jitter: Optional[float] = None,
) -> torch.Tensor:
    """Generate a grid of pixel coordinates.

    Args:
        resolution: the resolution of the grid, in (height, width).
        batch_size: the batch size of the output.
        subsample_to: if set, the number of subsampled points in the output.
        invert_y_axis: whether to invert the y-axis.
        margin: the margin around the grid.
        corner_aligned: whether to align the corners of the grid.
            If False, the value range will be generated with range. If True, the value range will be generated with linspace.
        jitter: add additional jitter to the grid values.

    Returns:
        a tensor of shape (batch_size, n_points, 2) containing the pixel coordinates.
    """
    h, w = resolution
    n_points = resolution[0] * resolution[1]
    uh = 1 if align_corners else 1 - (1 / h)
    uw = 1 if align_corners else 1 - (1 / w)
    if margin > 0:
        uh = uh + (2 / h) * margin
        uw = uw + (2 / w) * margin
        w, h = w + margin * 2, h + margin * 2

    x, y = torch.linspace(-uw, uw, w), torch.linspace(-uh, uh, h)
    if jitter is not None:
        dx = (torch.ones_like(x).uniform_() - 0.5) * 2 / w * jitter
        dy = (torch.ones_like(y).uniform_() - 0.5) * 2 / h * jitter
        x, y = x + dx, y + dy
    x, y = torch.meshgrid(x, y)
    pixel_scaled = torch.stack([x, y], -1).permute(1, 0, 2).reshape(1, -1, 2).repeat(batch_size, 1, 1)

    if subsample_to is not None and subsample_to > 0 and subsample_to < n_points:
        idx = np.random.choice(pixel_scaled.shape[1], size=(subsample_to,), replace=False)
        pixel_scaled = pixel_scaled[:, idx]

    if invert_y_axis:
        pixel_scaled[..., -1] *= -1.0

    return pixel_scaled


def get_features_per_vertex(
    mesh: Union[MeshContainer, Meshes],
    device: Optional[str] = None, diffusion_pipeline: Optional[nn.Module] = None, dino_model: Optional[nn.Module] = None, *,
    mesh_vertices: Optional[torch.Tensor] = None,
    prompt: Optional[str] = None, prompts_list: Optional[List[str]] = None,
    num_views: int = 100, H: int = 512, W: int = 512,
    use_latent: bool = False, use_normal_map: bool = True,
    use_ball_query: bool = True, ball_query_radius_factor: float = 0.01,
    num_images_per_prompt: int = 1, return_image: bool = True,
    verbose: bool = False,
):
    """
    Extract features per vertex from a mesh using a diffusion model and a DINO model. This function has three steps:

    1. Render the mesh from multiple views.
    2. Extract features from the rendered images.
        - Use a diffusion model to add textures to the rendered images.
        - Use a DINO model to extract features from the rendered images.
        - Combine the features from the diffusion model and the DINO model for each pixel.
        - Map the features back to the vertices of the mesh using a ball query or the nearest neighbor (depending on `use_ball_query`).
    3. Aggregate the features per vertex across the rendered views.

    Args:
        mesh: the mesh from which features will be extracted.
        device: the device to use for computation. defaults to 'cuda' if available, otherwise 'cpu'.
        diffusion_pipeline: the diffusion model used to generate features.
        dino_model: the DINO model used to extract features.
        mesh_vertices: the vertices where features will be extracted. if not provided, the mesh vertices will be used.
        prompt: the prompt used to generate texture completions for the diffusion pipeline.
        prompts_list: the list of prompts used to generate texture completions for the diffusion pipeline.
        num_views: the number of views to use for feature extraction.
        H: the height of the rendered images.
        W: the width of the rendered images.
        use_latent: whether to use latent diffusion in the diffusion pipeline (not implemented yet).
        use_normal_map: whether to use normal maps in the diffusion pipeline (not implemented yet).
        use_ball_query: whether to use ball queries to map features back to the vertices of the mesh.
        ball_query_radius_factor: the radius of the ball query. The radius is computed as a `f * maximal_distance_between_vertices`.
        num_images_per_prompt: the number of images to generate per prompt.
        return_image: whether to return the generated images.
        verbose: whether to print verbose output such as the number of missing features and the runtime.
    """
    t1 = time.time()

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if isinstance(mesh, MeshContainer):
        mesh = mesh.to_pytorch3d_meshes(device)

    if mesh_vertices is None:
        mesh_vertices = mesh.verts_list()[0]
    maximal_distance = get_maximal_distance(mesh_vertices)

    ball_drop_radius = maximal_distance * ball_query_radius_factor
    batched_renderings, normal_batched_renderings, camera, depth = batch_render(device, mesh, num_views, H, W, use_normal_map)

    if use_normal_map:
        normal_batched_renderings = normal_batched_renderings.cpu()

    batched_renderings = batched_renderings.cpu()
    camera = camera.cpu()
    depth = depth.cpu()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Generate a grid of pixel coordinates
    pixel_coords = arange_pixels((H, W), invert_y_axis=True)[0]
    pixel_coords[:, 0] = torch.flip(pixel_coords[:, 0], dims=[0])
    grid = arange_pixels((H, W), invert_y_axis=False)[0].to(device).reshape(1, H, W, 2).half()

    ft_per_vertex = None
    ft_per_vertex_count = torch.zeros((len(mesh_vertices), 1)).half()  # .to(device)
    normal_map_input = None

    for idx in tqdm(range(len(batched_renderings))):
        dp = depth[idx].flatten().unsqueeze(1)  # (H*W, 1)
        xy_depth = torch.cat((pixel_coords, dp), dim=1)  # (H*W, 3)

        # Filter out invalid depth values
        indices = xy_depth[:, 2] != -1
        xy_depth = xy_depth[indices]

        # Unproject pixel coordinates to world coordinates: (H*W, 3) -> (H*W, 3)
        world_coords = camera[idx].unproject_points(xy_depth, world_coordinates=True, from_ndc=True).to(device)

        rendered_image = (batched_renderings[idx, :, :, :3].cpu().numpy() * 255).astype(np.uint8)

        aligned_diffusion_features = None
        diffusion_output = None
        if diffusion_pipeline is not None:
            raise NotImplementedError(f'Diffusion features are not implemented yet.')

            diffusion_input_img = rendered_image
            if use_normal_map:
                normal_map_input = normal_batched_renderings[idx]
            depth_map = depth[idx, :, :, 0].unsqueeze(0).to(device)
            if prompts_list is not None:
                prompt = random.choice(prompts_list)
            diffusion_output = add_texture_to_render(
                diffusion_pipeline, diffusion_input_img, depth_map, prompt,
                normal_map_input=normal_map_input, use_latent=use_latent, num_images_per_prompt=num_images_per_prompt, return_image=return_image
            )

            with torch.no_grad():
                ft = torch.nn.Upsample(size=(H,W), mode="bilinear")(diffusion_output[0].unsqueeze(0)).to(device)
                ft_dim = ft.size(1)
                aligned_diffusion_features = torch.nn.functional.grid_sample(ft, grid, align_corners=False).reshape(1, ft_dim, -1)
                aligned_diffusion_features = torch.nn.functional.normalize(aligned_diffusion_features, dim=1)

        aligned_dino_features = None
        if dino_model is not None:
            if diffusion_output is not None:
                dino_input_img = diffusion_output[1][0]
            else:
                dino_input_img = Image.fromarray(rendered_image)
            aligned_dino_features = get_dino_features(device, dino_model, dino_input_img, grid)

        features = list(filter(lambda x: x is not None, [aligned_diffusion_features, aligned_dino_features]))
        if len(features) == 0:
            raise ValueError("No features extracted")
        aligned_features = torch.hstack(features) / len(features)

        if ft_per_vertex is None:
            ft_per_vertex = torch.zeros((len(mesh_vertices), aligned_features.shape[1]), dtype=aligned_features.dtype, device=device)
        features_per_pixel = aligned_features[0, :, indices].cpu()

        if use_ball_query:
            queried_indices = ball_query(world_coords.unsqueeze(0), mesh_vertices.unsqueeze(0), K=100, radius=ball_drop_radius, return_nn=False).idx[0].cpu()
            mask = queried_indices != -1
            repeat = mask.sum(dim=1)
            ft_per_vertex_count[queried_indices[mask]] += 1
            ft_per_vertex[queried_indices[mask]] += features_per_pixel.repeat_interleave(repeat, dim=1).T
        else:
            distances = torch.cdist(world_coords, mesh_vertices, p=2)
            closest_vertex_indices = torch.argmin(distances, dim=1).cpu()
            ft_per_vertex[closest_vertex_indices] += features_per_pixel.T
            ft_per_vertex_count[closest_vertex_indices] += 1

    idxs = (ft_per_vertex_count != 0)[:, 0]
    ft_per_vertex[idxs, :] = ft_per_vertex[idxs, :] / ft_per_vertex_count[idxs, :]
    missing_features = (ft_per_vertex_count == 0).sum().item()

    if verbose:
        print("Number of missing features: ", missing_features)
        print("Copied features from nearest vertices")

    if missing_features > 0:
        filled_indices = ft_per_vertex_count[:, 0] != 0
        missing_indices = ft_per_vertex_count[:, 0] == 0
        distances = torch.cdist(
            mesh_vertices[missing_indices], mesh_vertices[filled_indices], p=2
        )
        closest_vertex_indices = torch.argmin(distances, dim=1).cpu()
        ft_per_vertex[missing_indices, :] = ft_per_vertex[filled_indices][closest_vertex_indices, :]

    if verbose:
        t2 = time.time() - t1
        t2 = t2 / 60
        print("Time taken in mins: ", t2)

    return ft_per_vertex
