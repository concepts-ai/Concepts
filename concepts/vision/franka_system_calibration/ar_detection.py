#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : ar_detection.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/01/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Tuple, List, Dict, NamedTuple

import cv2
import numpy as np

from concepts.utils.typing_utils import Vec2f, Vec3f, Vec4f


class ARTagDetection(NamedTuple):
    id: int
    center: Vec2f
    corners: List[Vec2f]


def get_ar_tag_detections(img: np.ndarray) -> Dict[int, ARTagDetection]:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, _ = detector.detectMarkers(img)
    if ids is None:
        return dict()

    rv = dict()
    for i, id in enumerate(ids):
        id = int(id)
        rv[id] = ARTagDetection(id, np.mean(corners[i][0], axis=0).tolist(), corners[i][0].tolist())
    return rv


def visualize_ar_tag_detections(img: np.ndarray, det: Dict[int, ARTagDetection], show: bool = True) -> np.ndarray:
    img = img.copy()
    for id, tag in det.items():
        uv = np.round(tag.center).astype(np.int32)
        cv2.circle(img, tuple(uv), 5, (0, 255, 0), -1)
        cv2.putText(img, str(id), tuple(uv), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        for i in range(4):
            cv2.line(img, tuple(np.round(tag.corners[i]).astype(np.int32)), tuple(np.round(tag.corners[(i + 1) % 4]).astype(np.int32)), (0, 255, 0), 2)

    if show:
        cv2.imshow('AR Tag UV', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img


def get_ar_tag_pose(det: ARTagDetection, ar_tag_size: float, intrinsics: np.ndarray) -> Tuple[Vec3f, Vec4f]:
    corners = np.array(det.corners)

    tag_size = ar_tag_size
    tag_points = np.array([
        [-tag_size / 2, -tag_size / 2, 0],
        [tag_size / 2, -tag_size / 2, 0],
        [tag_size / 2, tag_size / 2, 0],
        [-tag_size / 2, tag_size / 2, 0]
    ])
    _, rvec, tvec = cv2.solvePnP(tag_points, corners, intrinsics, None)
    rvec, _ = cv2.Rodrigues(rvec)
    tvec = tvec.flatten()

    return tvec, rvec


def get_transform_matrix(tvec: Vec3f, rvec: Vec4f) -> np.ndarray:
    rv = np.eye(4)
    rv[:3, :3] = rvec
    rv[:3, 3] = tvec
    return rv


def get_ar_tag_poses_from_camera_pose(det: Dict[int, ARTagDetection], camera_intrinsics: np.ndarray, camera_pose: np.ndarray) -> Dict[int, np.ndarray]:
    rv = dict()
    for id, tag in det.items():
        tvec, rvec = get_ar_tag_pose(tag, 0.1, camera_intrinsics)
        matrix = get_transform_matrix(tvec, rvec)
        rv[id] = np.dot(camera_pose, matrix)

    return rv

