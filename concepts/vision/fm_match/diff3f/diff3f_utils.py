#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : diff3f_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/15/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
import open3d as o3d
import torch
import random
from scipy.optimize import linear_sum_assignment

from jacinle.utils.tqdm import tqdm

__all__ = ['cosine_similarity', 'cosine_similarity_batch', 'hungarian_correspondence', 'get_ball_query_radius', 'get_maximal_distance']


VERTEX_GPU_LIMIT = 35000


def get_maximal_distance(mesh_vertices: torch.Tensor):
    """Get the maximal distance between the mesh vertices."""
    if len(mesh_vertices) > VERTEX_GPU_LIMIT:
        samples = random.sample(range(len(mesh_vertices)), 10000)
        return torch.cdist(mesh_vertices[samples], mesh_vertices[samples]).max()
    else:
        return torch.cdist(mesh_vertices, mesh_vertices).max()


def get_ball_query_radius(mesh_vertices: torch.Tensor, ball_query_radius_factor: float = 0.01):
    """Get the ball query radius."""
    return get_maximal_distance(mesh_vertices) * ball_query_radius_factor


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute the cosine similarity between two tensors.

    Args:
        a: A tensor of shape (N, D).
        b: A tensor of shape (M, D).

    Returns:
        A tensor of shape (N, M) containing the cosine similarity between each pair of vectors.
    """
    assert a.dim() == 2 and b.dim() == 2, 'Only support 2D tensors.'

    if len(a) > 30000:
        return cosine_similarity_batch(a, b, batch_size=30000)

    dot_product = torch.mm(a, b.t())
    norm_a = torch.norm(a, dim=1, keepdim=True)
    norm_b = torch.norm(b, dim=1, keepdim=True)
    similarity = dot_product / (norm_a * norm_b.t())

    return similarity


def cosine_similarity_batch(a: torch.Tensor, b: torch.Tensor, batch_size: int = 30000) -> torch.Tensor:
    """Compute the cosine similarity between two tensors in a batch manner.

    Args:
        a: A tensor of shape (N, D).
        b: A tensor of shape (M, D).
        batch_size: The batch size.

    Returns:
        A tensor of shape (N, M) containing the cosine similarity between each pair of vectors.
    """

    num_a, _ = a.size()
    num_b, _ = b.size()

    similarity_matrix = torch.empty(num_a, num_b, device="cpu")
    for i in tqdm(range(0, num_a, batch_size)):
        a_batch = a[i:i+batch_size]
        for j in range(0, num_b, batch_size):
            b_batch = b[j:j+batch_size]
            dot_product = torch.mm(a_batch, b_batch.t())
            norm_a = torch.norm(a_batch, dim=1, keepdim=True)
            norm_b = torch.norm(b_batch, dim=1, keepdim=True)
            similarity_batch = dot_product / (norm_a * norm_b.t())
            similarity_matrix[i:i+batch_size, j:j+batch_size] = similarity_batch.cpu()

    return similarity_matrix


def hungarian_correspondence(similarity_matrix: torch.Tensor) -> torch.Tensor:
    """Compute the hungarian correspondence based on a similarity matrix.

    Args:
        similarity_matrix: A tensor of shape (N, M) containing the cosine similarity between each pair of vectors.

    Returns:
        A tensor of shape (N, M) containing the hungarian correspondence between each pair of vectors.
    """
    # Convert similarity matrix to a cost matrix by negating the similarity values
    cost_matrix = -similarity_matrix.cpu().numpy()

    # Use the Hungarian algorithm to find the best assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Create a binary matrix with 1s at matched indices and 0s elsewhere
    num_rows, num_cols = similarity_matrix.shape
    match_matrix = np.zeros((num_rows, num_cols), dtype=int)
    match_matrix[row_indices, col_indices] = 1
    match_matrix = torch.from_numpy(match_matrix).to(similarity_matrix.device)
    return match_matrix


def project_features_on_pointcloud(mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud, feature: torch.Tensor, ball_query_radius_factor: float = 0.1):
    """Project the features on the pointcloud."""
    pcd_points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32)

    vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32)

    ball_query_radius = get_ball_query_radius(vertices, ball_query_radius_factor)

    from pytorch3d.ops import ball_query

    rv = ball_query(pcd_points.unsqueeze(0), vertices.unsqueeze(0), K=20, radius=ball_query_radius, return_nn=False)
    queried_distances = rv.dists[0].cpu()
    queried_indices = rv.idx[0].cpu()

    invalid_mask = queried_indices < 0
    queried_indices[invalid_mask] = 0
    queried_distances[invalid_mask] = 1

    queried_features = feature[queried_indices]  # (N, K, C)
    queried_features = queried_features / queried_distances.unsqueeze(dim=-1)
    queried_features[invalid_mask] = 0
    normalized_queried_features = queried_features.sum(dim=1) / (~invalid_mask).sum(dim=1).unsqueeze(dim=-1).clamp(min=1)

    return normalized_queried_features
