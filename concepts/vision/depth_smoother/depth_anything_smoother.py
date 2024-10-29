#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : depth_anything_smoother.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/30/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Use the Depth Anything v2 model to smooth the depth map."""

import cv2
import numpy as np
import numpy.random as npr
import transformers
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import Optional, NamedTuple
from PIL import Image
from scipy.interpolate import UnivariateSpline
from sklearn.cluster import KMeans


class DepthAnythingV2SmootherConfig(NamedTuple):
    model_size: str = 'small'
    """The size of the Depth Anything V2 model. Options are 'small', 'medium', 'large'."""

    mode: str = 'multi-cubic'
    """The mode of the spline fitting. Options are 'linear', 'multi-linear', 'mulit-cubic', 'cluster-cubic'."""

    remove_edge: bool = True
    """Whether to remove the edge of the predicted depth map. These areas are usually noisy."""

    curve_smooth: float = 1
    """The smoothness of the spline fitting. Only used when mode is 'cubic' or 'cluster-cubic'."""

    nr_samples: int = 10
    """The number of samples to take for the spline fitting."""

    nr_sample_points: int = 100
    """The number of points per sample to take for each sample."""

    @classmethod
    def make_default(cls):
        return cls()


class DepthAnythingV2Smoother(object):
    """The Depth Anything V2 smoother.

    The Depth Anything V2 model is a depth estimation model that can be used to smooth the depth map.
    In particular, we will first use the Depth Anything V2 model to generate a predicted depth map of the input RGB image.

    The Depth Anything V2 model is available in three sizes: 'small', 'medium', 'large'.

    Then, we will fit splines between the observed depth map and the predicted depth map. The spline fitting can be done in multiple ways:

    - 'linear': fit a single line to the depth map using the predicted depth map.
    - 'multi-linear': fit multiple lines to the depth map using the predicted depth map. We will take the median of the predicted depth map.
    - 'multi-cubic': fit multiple cubic splines to the depth map using the predicted depth map. We will take the median of the predicted depth map.
    - 'cluster-cubic': fit cubic splines to the depth map using the predicted depth map. We will cluster the depth map into several clusters and fit a spline to each cluster.

    Finally, we will use the fitted splines to transform the predicted depth map to a smoothed depth map.
    """
    MODEL_SIZE = {
        'small': 'depth-anything/Depth-Anything-V2-Small-hf',
        'medium': 'depth-anything/Depth-Anything-V2-Medium-hf',
        'large': 'depth-anything/Depth-Anything-V2-Large-hf',
    }

    def __init__(self, config: Optional[DepthAnythingV2SmootherConfig] = None, device: str = 'cpu'):
        """Initialize the Depth Anything V2 smoother.

        Args:
            config: the configuration of the smoother.
            device: the device to use for the model.
        """
        self.config = config if config is not None else DepthAnythingV2SmootherConfig.make_default()
        model_name = type(self).MODEL_SIZE[self.config.model_size]
        self.dam_pipe = transformers.pipeline(task="depth-estimation", model=model_name, device=device)

    def __call__(self, rgb: np.ndarray, depth: np.ndarray, mask: Optional[np.ndarray] = None, visualize: bool = False):
        """Smooth the depth map using the Depth Anything V2 model.

        Args:
            rgb: the RGB image.
            depth: the input depth map.
            mask: the mask of the valid pixels. This mask will used to sample points from the depth/predicted_depth map.
        """
        if depth.max() > 1000:
            raise ValueError('The input depth map is not in meters. Expect the depth map to be < 1000.')

        rv = self.dam_pipe(Image.fromarray(rgb))
        predicted_depth = np.asarray(rv['predicted_depth'][0])
        # NB(Jiayuan Mao @ 2024/07/30): We need to manually resize the predicted depth map to the same size as the input depth map.
        #     The default behavior of the Depth Anything V2 model is to resize the depth prediction using Bilinear interpolation,
        #     which will introduce artifacts in the depth map.
        predicted_depth = F.interpolate(
            torch.tensor(predicted_depth[None, None]), size=depth.shape, mode='nearest', align_corners=None
        ).squeeze().numpy()

        # Remove the "edges" of the depth map.
        edge_mask = None
        if self.config.remove_edge:
            predicted_depth, edge_mask = remove_edge(predicted_depth)

        if self.config.mode == 'linear':
            return _remove_values(fit_single_line(depth, predicted_depth), edge_mask)
        elif self.config.mode == 'multi-linear' or self.config.mode == 'multi-cubic':
            depth_flat = depth.flatten()
            predicted_depth_flat = predicted_depth.flatten()
            mask = mask.flatten() if mask is not None else None

            all_samples = []
            for i in range(self.config.nr_samples):
                _, ds, predicted_ds = sample_and_retrieve_corresponding_depths_flat(depth_flat, predicted_depth_flat, self.config.nr_sample_points, mask)


                if self.config.mode == 'multi-cubic':
                    spline, sorted_ds, sorted_predicted_ds = fit_spline_ransac(ds, predicted_ds, s=self.config.curve_smooth, return_sorted_points=True)
                    new_predicted_depth_flat = spline(predicted_depth_flat)

                    if visualize:
                        plt.plot(predicted_ds, ds, 'o')
                        plt.plot(sorted_predicted_ds, spline(sorted_predicted_ds), '-', c='r')
                        plt.show()
                else:
                    new_predicted_depth_flat = fit_single_line(ds, predicted_ds, predicted_depth_flat)
                    if visualize:
                        plt.plot(predicted_ds, ds, 'o', c='r')
                        plt.plot(predicted_depth_flat, new_predicted_depth_flat, 'o', c='g')
                        plt.show()

                all_samples.append(new_predicted_depth_flat)

            all_samples = np.stack(all_samples, axis=-1)
            all_samples_median = np.median(all_samples, axis=-1)

            # Select top 3 values that are close to the median
            top_k = 3
            indices = np.argsort(np.abs(all_samples - all_samples_median[:, None]), axis=1)
            indices = indices[:, :top_k]
            all_samples = all_samples[np.arange(all_samples.shape[0])[:, None], indices]
            # Take the mean
            final_depth = np.mean(all_samples, axis=-1)
            return _remove_values(final_depth.reshape(depth.shape), edge_mask)
        elif self.config.mode == 'cluster-cubic':
            depth_flat = depth.flatten()
            predicted_depth_flat = predicted_depth.flatten()
            mask = mask.flatten() if mask is not None else None

            _, labels = clustering_depth(depth, predicted_depth, n_clusters=10)

            if visualize:
                fig, ax = plt.subplots(1, 2)
                visualize_indices = np.random.choice(np.arange(len(depth_flat)), 100)
                ax[0].scatter(predicted_depth_flat[visualize_indices], depth_flat[visualize_indices], c=labels[visualize_indices])
                ax[1].imshow(labels.reshape(depth.shape))
                plt.suptitle('Clustering Results')
                plt.show()

            new_predicted_depth = np.zeros_like(depth_flat)
            for i in range(labels.max()):
                this_mask = labels == i
                if mask is not None:
                    this_mask = this_mask & mask
                if this_mask.sum() < 100:
                    continue

                ds = depth_flat[this_mask]
                predicted_ds = predicted_depth_flat[this_mask]

                _, s_ds, s_predicted_ds = sample_and_retrieve_corresponding_depths_flat(ds, predicted_ds, self.config.nr_sample_points, this_mask)
                spline, sorted_ds, sorted_predicted_ds = fit_spline(s_ds, s_predicted_ds, s=self.config.curve_smooth, return_sorted_points=True)
                new_predicted_depth_flat = spline(predicted_depth_flat)

                if visualize:
                    plt.figure()
                    plt.plot(predicted_ds, ds, 'o')
                    plt.plot(sorted_predicted_ds, spline(sorted_predicted_ds), '-', c='r')
                    plt.title(f'Curve Fitting for Cluster{i}')
                    plt.show()

                # We will still update the depth map even if they are not part of the "mask".
                new_predicted_depth[labels == i] = new_predicted_depth_flat[labels == i]

            return _remove_values(new_predicted_depth.reshape(depth.shape), edge_mask)
        else:
            raise ValueError(f'Unknown mode: {self.config.mode}. Available options are "linear", "multi-linear", "multi-cubic", "cluster-cubic".')


def fit_single_line(depth: np.ndarray, predicted_depth: np.ndarray, target: Optional[np.ndarray] = None):
    """Fit a single line to the depth map using the predicted depth map.

    .. math::

        \begin{align*}
        \begin{bmatrix}
        predicted\_depth & 1
        \end{bmatrix}
        \begin{bmatrix}
        k \\
        b
        \end{bmatrix}
        = depth
        \end{align*}

    Args:
        depth: the observed depth map.
        predicted_depth: the predicted depth map.
        target: the target depth map to transform.

    Returns:
        the transformed depth map. If target is not None, we will return `k * target + b`. Otherwise, we will return `k * predicted_depth + b`.
    """

    depth_flat = depth.flatten()
    predicted_depth_flat = predicted_depth.flatten()

    mask = depth_flat > 0
    depth_flat = depth_flat[mask]
    predicted_depth_flat = predicted_depth_flat[mask]

    A = np.vstack([predicted_depth_flat, np.ones(len(predicted_depth_flat))]).T
    k, b = np.linalg.lstsq(A, depth_flat, rcond=None)[0]

    if target is not None:
        return k * target + b
    return k * predicted_depth + b


def fit_spline(depth: np.ndarray, predicted_depth: np.ndarray, k: int = 3, s: float = 10, return_sorted_points: bool = False):
    """Fit a spline to the depth map using the predicted depth map. In particular, we will fit `depth = f(predicted_depth)`.

    Args:
        depth: the observed depth map.
        predicted_depth: the predicted depth map.
        k: the degree of the spline.
        s: the smoothness of the spline.
        return_sorted_points: whether to return the sorted points of the spline.

    Returns:
        the fitted spline. If `return_sorted_points` is True, we will also return the sorted points of the depth map and the predicted depth map,
            which can be used for visualization.
    """
    sorted_indices = np.argsort(predicted_depth)
    depth = depth[sorted_indices]
    predicted_depth = predicted_depth[sorted_indices]

    # Remove duplicate values
    predicted_depth, indices = np.unique(predicted_depth, return_index=True)
    depth = depth[indices]

    spline = UnivariateSpline(predicted_depth, depth, k=k, s=s)

    if return_sorted_points:
        return spline, depth, predicted_depth
    return spline


def fit_spline_ransac(depth: np.ndarray, predicted_depth: np.ndarray, k: int = 3, s: float = 10, ransac_n: int = 10, ransac_init_ratio: float = 0.5, ransac_threshold: float = 0.1, return_sorted_points: bool = False):
    """Fit a spline to the depth map using the predicted depth map. In particular, we will fit `depth = f(predicted_depth)`."""
    sorted_indices = np.argsort(predicted_depth)
    depth = depth[sorted_indices]
    predicted_depth = predicted_depth[sorted_indices]

    # Remove duplicate values
    predicted_depth, indices = np.unique(predicted_depth, return_index=True)
    depth = depth[indices]

    spline = 0
    indices = npr.choice(len(predicted_depth), int(len(predicted_depth) * ransac_init_ratio), replace=False)
    for i in range(ransac_n):
        indices = np.sort(indices)
        nr_inliers = len(indices)
        spline = UnivariateSpline(predicted_depth[indices], depth[indices], k=k, s=s)
        residuals = np.abs(spline(predicted_depth) - depth)
        indices = np.where(residuals < ransac_threshold)[0]

        if len(indices) == nr_inliers:
            break

    if return_sorted_points:
        return spline, depth, predicted_depth
    return spline


def remove_edge(depth: np.ndarray):
    """Remove the edge of the depth map. We will use the Canny edge detector to detect the edge of the depth map.
    And then we will dilate the edge to create a mask. Finally, we will set the values of the edge to 0.

    Args:
        depth: the depth map.

    Returns:
        the processed depth map and the edge mask.
    """

    raw_depth = depth.copy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    edge = cv2.Canny((depth * 255).astype(np.uint8), 50, 200)
    # dilate the edge
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=3)
    edge = edge > 0

    raw_depth[edge] = 0
    return depth, edge


def sample_and_retrieve_corresponding_depths_flat(observed_depth_flat: np.ndarray, predicted_depth_flat: np.ndarray, n: int, mask: np.ndarray, remove_zero_values: bool = True):
    """Sample n points from the depth map and the predicted depth map, and return the corresponding values.

    Args:
        observed_depth_flat: the observed depth map.
        predicted_depth_flat: the predicted depth map.
        n: the number of samples to take.
        mask: the mask of the valid pixels. We will only sample points from the pixels that are valid.
        remove_zero_values: whether to remove the zero values in the observed depth map and the predicted depth map.
    """
    observed_depth_flat = observed_depth_flat.flatten()
    predicted_depth_flat = predicted_depth_flat.flatten()
    xs = np.arange(observed_depth_flat.shape[0])

    if remove_zero_values:
        non_zero_mask = (observed_depth_flat > 0) & (predicted_depth_flat > 0)
        mask = mask & non_zero_mask if mask is not None else non_zero_mask

    if mask is not None:
        xs = xs[mask]
        observed_depth_flat = observed_depth_flat[mask]
        predicted_depth_flat = predicted_depth_flat[mask]

    indices = npr.choice(len(xs), n, replace=False)
    return xs[indices], observed_depth_flat[indices], predicted_depth_flat[indices]


def clustering_depth(depth: np.ndarray, predicted_depth: np.ndarray, n_clusters: int = 10):
    """Cluster the depth map using the predicted depth map.

    Args:
        depth: the observed depth map.
        predicted_depth: the predicted depth map.
        n_clusters: the number of clusters to use.

    Returns:
        the KMeans model and the labels.
    """

    data = np.stack([depth.flatten(), predicted_depth.flatten()], axis=1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    return kmeans, kmeans.predict(data)


def _remove_values(depth, edge_mask):
    """Remove the edge of the depth map in place."""
    if edge_mask is not None:
        depth[edge_mask] = 0
    return depth