#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dino_match.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/15/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import gc
from typing import Optional, Union, List

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from PIL import Image
from skimage.feature import peak_local_max


def patch_match(
        source_feature: torch.Tensor,
        target_features: torch.Tensor,  # N, C, H, W
        source_xy: Union[np.ndarray, List, tuple],
        patch_size=15,
        return_match_scores=False,
        collect_memory: Optional[bool] = None,
        max_peaks=5,
) -> Union[
    tuple[List, np.ndarray, List[np.ndarray]],
    tuple[List, np.ndarray, List[np.ndarray], List[float]]
]:
    if collect_memory is None:
        collect_memory = torch.cuda.is_available()

    def maybe_collect_memory():
        if collect_memory:
            gc.collect()
            torch.cuda.empty_cache()

    num_targets = len(target_features)
    K = patch_size
    half_patch_size = (patch_size - 1) // 2

    x, y = int(np.round(source_xy[0])), int(np.round(source_xy[1]))

    src_ft = source_feature.unsqueeze(0)
    src_patch = src_ft[
        0, :,
        y - half_patch_size:y + half_patch_size + 1,
        x - half_patch_size:x + half_patch_size + 1
    ]  # C, K, K

    src_rotated_patches = []

    for angle in range(0, 360, 90):
        rotated_patch = TF.rotate(src_patch, angle, expand=False)
        # TODO: make the rotated_patch same size
        src_rotated_patches.append(rotated_patch)

    del src_ft
    del src_patch
    maybe_collect_memory()

    src_rotated_patches = [F.normalize(patch, dim=0) for patch in src_rotated_patches]  # 1, C, K, K

    best_match_scores = []
    best_match_yxs = []
    heatmaps = []

    maybe_collect_memory()

    for i in range(num_targets):
        trg_ft = F.normalize(target_features[i], dim=0).unsqueeze(0)  # 1, C, H, W

        score_maps = []
        K = K
        _, C, H, W = trg_ft.shape
        for src_rotated_patch in src_rotated_patches:
            padding = (K - 1) // 2
            padded_trg_ft = F.pad(trg_ft, (padding, padding, padding, padding), mode='constant', value=0)
            unfolded = F.unfold(padded_trg_ft, kernel_size=K, stride=1)
            unfolded = unfolded.view(1, C, K, K, -1)
            similarity = (src_rotated_patch.unsqueeze(-1) * unfolded).sum(dim=(1, 2, 3)) / (K * K)
            similarity_map = similarity.view(1, H, W).cpu().numpy()
            score_maps.append(similarity_map)
            del unfolded
            del padded_trg_ft
            maybe_collect_memory()
        score_map = np.concatenate(score_maps, axis=0).max(axis=0)  # HxW

        local_peaks = find_local_peaks(score_map, max_peaks=max_peaks)

        # max_yx = np.unravel_index(score_map.argmax(), score_map.shape)
        # best_match_scores.append(score_map.max())
        # best_match_yxs.append(max_yx)

        for peak_y, peak_x in local_peaks:
            best_match_scores.append(score_map[peak_y, peak_x])
            best_match_yxs.append((i, peak_y, peak_x))

        heatmap = score_map
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]

        heatmaps.append(heatmap)

        del score_map
        maybe_collect_memory()

    argsort = np.argsort(np.array(best_match_scores))
    match_order = np.argsort(argsort)
    if return_match_scores:
        return best_match_yxs, match_order, heatmaps, best_match_scores
    else:
        return best_match_yxs, match_order, heatmaps


def find_local_peaks(heatmap: np.ndarray, threshold: float = 0.0, min_distance: int = 5, max_peaks: int = 5):
    peaks = peak_local_max(heatmap, threshold_abs=threshold, min_distance=min_distance, num_peaks=max_peaks)
    return peaks


def visualize_match(
    source_img: Image.Image,
    target_imgs: list[Image.Image],
    source_xy: tuple[int, int],
    best_match_yxs: List[Union[np.ndarray, tuple]],
    best_match_scores: List[float],
    match_order: np.ndarray,
    heatmaps: List[np.ndarray] = None,
    num_rows: int = None,
    save_path: str = None,
    save_and_show: bool = False
):
    window_size = 5
    num_targets = len(target_imgs)
    if num_rows is None:
        num_rows = int(np.floor(np.sqrt(num_targets)))
    num_columns = int(np.ceil(num_targets / num_rows) + 1)
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(window_size * num_columns, window_size * num_rows))
    axs = np.atleast_2d(axs)
    for i in range(num_rows):
        axs[i][0].imshow(source_img)
        axs[i][0].scatter(source_xy[0], source_xy[1], c='r', s=100)
        axs[i][0].set_title(f'({source_xy[0]}, {source_xy[1]})', fontsize=25)
        axs[i][0].axis('off')

    for i in range(num_targets):
        row = i // (num_columns - 1)
        column = i % (num_columns - 1) + 1
        axs[row][column].imshow(target_imgs[i])
        if heatmaps is not None:
            axs[row][column].imshow(heatmaps[i], alpha=0.5)

        this_best_match_yxs = [
            (best_match_yx[1], best_match_yx[2], best_match_score)
            for best_match_yx, best_match_score in zip(best_match_yxs, best_match_scores)
            if best_match_yx[0] == i
        ]

        for y, x, score in this_best_match_yxs:
            axs[row][column].scatter(x, y, c='r', s=30)
            axs[row][column].annotate(f'{score:.2f}', (x, y - 0.03 * target_imgs[i].width), fontsize=10, color='r', ha='center', va='center')

        # if len(this_best_match_yxs) == 1:
        #     axs[row][column].set_title(f'{match_order[i]}, ({best_match_yx[1]}, {best_match_yx[0]})', fontsize=25)
        axs[row][column].axis('off')

    if save_path is not None:
        plt.savefig(save_path)
        if save_and_show:
            plt.show()
    else:
        plt.show()

    plt.close()


def visualize_matches(
    source_img: Image.Image,
    target_imgs: list[Image.Image],
    source_xys: List[tuple[int, int]],
    best_match_yxs_s: List[List[Union[np.ndarray, tuple]]],
    best_match_scores_s: List[List[float]],
    match_orders: List[np.ndarray],
    heatmaps: List[np.ndarray] = None,
    num_rows: int = None,
    save_path: str = None,
    save_and_show: bool = False
):
    window_size = 5
    num_targets = len(target_imgs)
    if num_rows is None:
        num_rows = int(np.floor(np.sqrt(num_targets)))
    num_columns = int(np.ceil(num_targets / num_rows) + 1)
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(window_size * num_columns, window_size * num_rows))
    axs = np.atleast_2d(axs)
    for p in range(len(source_xys)):
        source_xy = source_xys[p]
        best_match_yxs = best_match_yxs_s[p]
        best_match_scores = best_match_scores_s[p]
        match_order = match_orders[p]
        if p == 0:
            color = 'r'
        elif p == 1:
            color = 'g'
        else:
            color = 'b'
        for i in range(num_rows):
            if p == 0:
                axs[i][0].imshow(source_img)
            axs[i][0].scatter(source_xy[0], source_xy[1], c=color, s=100)
            axs[i][0].set_title(f'({source_xy[0]}, {source_xy[1]})', fontsize=25)
            axs[i][0].axis('off')

        for i in range(num_targets):
            row = i // (num_columns - 1)
            column = i % (num_columns - 1) + 1
            axs[row][column].imshow(target_imgs[i])
            if heatmaps is not None:
                axs[row][column].imshow(heatmaps[i], alpha=0.5)

            this_best_match_yxs = [
                (best_match_yx[1], best_match_yx[2], best_match_score, order)
                for best_match_yx, best_match_score, order in zip(best_match_yxs, best_match_scores, match_order)
                if best_match_yx[0] == i
            ]
            this_best_match_yxs.sort(key=lambda x: x[2], reverse=True)
            # import ipdb; ipdb.set_trace()
            for y, x, score, order in this_best_match_yxs[:2]:
                axs[row][column].scatter(x, y, c=color, s=30)
                axs[row][column].annotate(f'{score:.2f}', (x, y - 0.03 * target_imgs[i].width), fontsize=10, color=color, ha='center', va='center')

            # if len(this_best_match_yxs) == 1:
            #     axs[row][column].set_title(f'{match_order[i]}, ({best_match_yx[1]}, {best_match_yx[0]})', fontsize=25)
            axs[row][column].axis('off')

    if save_path is not None:
        plt.savefig(save_path)
        if save_and_show:
            plt.show()
    else:
        plt.show()

    plt.close()

