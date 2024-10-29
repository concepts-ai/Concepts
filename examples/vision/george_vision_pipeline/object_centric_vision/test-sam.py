#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test-sam.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/05/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import jacinle
import numpy as np
import cv2
from segment_anything import SamPredictor, sam_model_registry

pickle = jacinle.load('./data-000001.pkl')
img = pickle['color_image'][..., ::-1]

sam = sam_model_registry["default"](checkpoint="./sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
predictor.set_image(img)

pos = (1136, 471)

def record_pos(event, x, y, flags, param):
    global pos
    if event == cv2.EVENT_LBUTTONDOWN:
        pos = (x, y)


def remove_remains(img, interest_point):
    """
    Remove remains which are not adjacent with interest_point
    :param img: Input image
    :param interest_point: Center point where we want to remain
    :return: Image which adjacent with interest_point
    """
    img = img.astype(np.uint8)
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    img_inv = img.copy()
    cv2.floodFill(img_inv, mask, tuple(interest_point), 0)
    img -= img_inv

    return img


while True:
    print('Runing at', pos)
    masks, _, _ = predictor.predict(point_coords=np.array([pos]), point_labels=np.array([1]))
    print('Number of masks:', len(masks))
    visualize = img.copy()[..., ::-1]
    mask = masks[-2].astype(np.uint8)
    mask = remove_remains(mask, pos)

    overlay = (mask > 0).astype(np.uint8)[:, :, None] * np.array([0, 255, 0], dtype=np.uint8)
    visualize = cv2.addWeighted(visualize, 1, overlay.astype('uint8'), 0.5, 0)

    cv2.circle(visualize, pos, 10, (255, 255, 0), -1)
    cv2.imshow('image', visualize)
    cv2.imshow('mask', overlay.astype('uint8'))
    cv2.setMouseCallback('image', record_pos)
    if cv2.waitKey(-1) & 0xFF == ord('q'):
        break

