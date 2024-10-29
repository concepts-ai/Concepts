#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test-ar-tag-detection.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/01/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import cv2
import numpy as np

import jacinle
from concepts.vision.franka_system_calibration.ar_detection import get_ar_tag_detections, visualize_ar_tag_detections, get_ar_tag_poses_from_camera_pose


def test_ar_tag_detection():
    parser = jacinle.JacArgumentParser()
    parser.add_argument('--image', required=True)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    dets = get_ar_tag_detections(img)
    print(dets)

    visualize_ar_tag_detections(img, dets)


def test_ar_tag_pose_estimation():
    parser = jacinle.JacArgumentParser()
    parser.add_argument('--image', required=True)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    dets = get_ar_tag_detections(img)
    print(dets)

    poses = get_ar_tag_poses_from_camera_pose(dets, camera_intrinsics=np.eye(3), camera_pose=np.eye(4))
    print(poses)


if __name__ == '__main__':
    # test_ar_tag_detection()
    test_ar_tag_pose_estimation()