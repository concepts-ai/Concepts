#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : shapes_detection_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/18/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import cv2
import numpy as np
import torch

__all__ = ['detect_shapes', 'tensor_to_image', 'image_to_tensor']


def detect_shapes(image):
    output = list()
    patches = _split_image_into_objects(image)
    for x in patches:
        if _detect_object(x):
            output.append((_detect_size(x), _detect_color(x), _detect_shape(x)))
        else:
            output.append(None)
    return patches, output


def tensor_to_image(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().numpy()
    x = x.transpose(1, 2, 0)
    x = (x * 0.5 + 0.5) * 255
    x = x.clip(0, 255).astype(np.uint8)
    return x


def image_to_tensor(x: np.ndarray) -> torch.Tensor:
    x = x.transpose(2, 0, 1)
    x = x / 255.0 * 2 - 1
    x = torch.from_numpy(x).float()
    return x


def _split_image_into_objects(x):
    x = x.reshape((3, 10, 3, 10, 3))
    x = x.transpose((0, 2, 1, 3, 4))
    return [x[i, j] for i in range(3) for j in range(3)]


def _detect_object(x):
    if (x > 5).sum():
        return True
    return False


def _detect_color(x):
    x = x.reshape(-1, 3)
    x = x[x.max(-1) > 5]
    x = x.mean(axis=0)
    c = x.argmax()
    return ['red', 'green', 'blue'][c]


def _detect_size(x):
    x_shape = x.shape
    x = x.reshape(-1, 3)
    x = x.max(-1) > 5
    x = x.reshape(x_shape[:2])
    x_range = x.any(axis=1).nonzero()
    y_range = x.any(axis=0).nonzero()
    boundary = (x_range[0].max() - x_range[0].min() + 1, y_range[0].max() - y_range[0].min() + 1)
    if max(boundary) > 7:
        return 'big'
    return 'small'


def _detect_shape(x):
    gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (256, 256), cv2.INTER_NEAREST)
    _, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]

    approx = cv2.approxPolyDP(contour, 0.1 * cv2.arcLength(contour, True), True)
    if len(approx) == 3:
        return 'triangle'

    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

    if len(approx) == 4:
        return 'square'
    else:
        return 'circle'

