#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : gpt_image_query_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/12/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Utility functions for preparing image queries for GPT models, such as drawing bounding boxes or drawing grids on images."""

from typing import Optional, Tuple, List

import cv2
import seaborn as sns
import numpy as np


def resize_to(img: np.ndarray, target_max_dim: int) -> np.ndarray:
    max_dim = max(img.shape[:2])
    scale = target_max_dim / max_dim
    target_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    return cv2.resize(img, target_size)


def draw_text_inplace(
    img_: np.ndarray, text: str, x: int, y: int,
    font_scale: float = 2, font_thickness: int = 3, text_color: Tuple[int, ...] = (0, 0, 0), bg_color: Optional[Tuple[int, ...]] = None, font = cv2.FONT_HERSHEY_PLAIN
) -> None:
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size

    if bg_color is not None:
        cv2.rectangle(img_, (x, y), (x + text_w, y + text_h), bg_color, -1)

    cv2.putText(img_, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)


def draw_text(img: np.ndarray, text: str, x: int, y: int, font_scale: float = 2, font_thickness: int = 3, text_color: Tuple[int, ...] = (0, 0, 0), bg_color: Optional[Tuple[int, ...]] = None, font = cv2.FONT_HERSHEY_PLAIN) -> np.ndarray:
    img = img.copy()
    draw_text_inplace(img, text, x, y, font_scale, font_thickness, text_color, bg_color, font)
    return img


def draw_grid(img: np.ndarray, nr_vertical: int, nr_horizontal: int, resize_to_max_dim: int = 0) -> np.ndarray:
    """Draw a grid on the image with nr_vertical and nr_horizontal lines. It will also put a number at the top-left corner of each cell."""

    if resize_to_max_dim > 0:
        img = resize_to(img, resize_to_max_dim)

    img = img.copy()
    h, w = img.shape[:2]
    for i in range(1, nr_vertical):
        x = i * w // nr_vertical
        cv2.line(img, (x, 0), (x, h), (255, 255, 255), 3)
    for i in range(1, nr_horizontal):
        y = i * h // nr_horizontal
        cv2.line(img, (0, y), (w, y), (255, 255, 255), 3)

    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 2
    font_thickness = 3
    text_color = (0, 0, 0)
    text_color_bg = (255, 255, 255)

    for i in range(nr_horizontal):
        for j in range(nr_vertical):
            x = j * w // nr_vertical +  w // nr_vertical // 2
            y = i * h // nr_horizontal + h // nr_horizontal // 2

            text = f"{i * nr_vertical + j + 1}"
            draw_text_inplace(img, text, x, y, font_scale, font_thickness, text_color, text_color_bg, font)

    return img


def draw_masks(img: np.ndarray, masks: List[np.ndarray], alpha: float = 0.5, contour_width: int = 2, mode: str = 'mask+bbox', resize_to_max_dim: int = 0) -> np.ndarray:
    """Draw masks on the image with the specified color and alpha value."""

    if resize_to_max_dim > 0:
        img = resize_to(img, resize_to_max_dim)

    draw_mask = 'mask' in mode
    draw_contour = 'contour' in mode
    draw_bbox = 'bbox' in mode

    img = img.copy()
    nr_colors = len(masks)
    colors = sns.color_palette('bright', n_colors=nr_colors)
    for i, mask in enumerate(masks):
        if resize_to_max_dim > 0:
            mask = (resize_to(mask, resize_to_max_dim) > 0.5)

        text_pos = None
        contour_color = tuple(map(int, np.array(colors[i][:3]) * 255))

        if draw_mask:
            img = img.astype(np.float32)
            img[mask > 0] = img[mask > 0] * alpha + np.array(contour_color) / np.array(contour_color).max() * (1 - alpha) * 255
            img = img.astype(np.uint8)
            x = np.where(mask > 0)[1]
            y = np.where(mask > 0)[0]
            text_pos = (int(x.mean()) + 5, int(y.mean()) + 5)

        if draw_contour:
            contours, _ = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            contours_valid = [x for x in contours if cv2.contourArea(x) > 50]
            if len(contours_valid) > 0:
                contour = contours_valid[0]
            else:
                contour = None

            cv2.drawContours(img, contour, -1, contour_color, thickness=contour_width)

            if contour is not None:
                M = cv2.moments(contour)
                text_pos = (round(M['m10'] / M['m00']), round(M['m01'] / M['m00']))

        if draw_bbox:
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            cv2.rectangle(img, (x, y), (x + w, y + h), contour_color, contour_width)

        if text_pos is not None:
            draw_text_inplace(img, str(i), text_pos[0], text_pos[1], font_scale=1, font_thickness=1, text_color=(0, 0, 0), bg_color=(255, 255, 255))

    return img

