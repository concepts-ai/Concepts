#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : extractor_dino.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/16/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Union
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as tfs

patch_size = 14


def init_dino(device: str) -> torch.nn.Module:
    """Initialize the DINO model.

    Args:
        device: the device to use.

    Returns:
        the DINO model.
    """
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model = model.to(device).eval()
    return model


@torch.no_grad()
def get_dino_features(device: str, dino_model: torch.nn.Module, img: Union[Image.Image, np.ndarray], grid: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Get the DINO features for a given image and grid.

    This function will always resize the image to (518, 518) and then compute its DINO features. This will result in a feature map of (37, 37).
    Then, we will "project" the feature map to the grid using bilinear interpolation.

    Args:
        device: the device to use.
        dino_model: the DINO model.
        img: the image to extract features from.
        grid: the grid to project the features to.
        normalize: whether to normalize the features.

    Returns:
        the grid-projected DINO features.
    """
    transform = tfs.Compose([
        tfs.Resize((518, 518)),
        tfs.ToTensor(),
        tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)[:3].unsqueeze(0).to(device)
    features = dino_model.get_intermediate_layers(img, n=1)[0].half()
    h, w = int(img.shape[2] / patch_size), int(img.shape[3] / patch_size)
    dim = features.shape[-1]
    features = features.reshape(-1, h, w, dim).permute(0, 3, 1, 2)

    if device == 'cpu':
        # NB(Jiayuan Mao @ 2024/09/16): When using CPU, some operations such as grid_sample are not supported for half precision.
        features = features.to(torch.float32)
        grid = grid.to(torch.float32)

    features = F.grid_sample(features, grid, align_corners=False).reshape(1, 768, -1)
    if normalize:
        features = F.normalize(features, dim=1)
    return features
