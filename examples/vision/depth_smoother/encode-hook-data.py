#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : encode-scooper-data.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/31/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp

import cv2
import numpy as np
import PIL.Image as Image
import jacinle.io as io


def load_rgbd(base_path, index, camera_index=0):
    rgb = Image.open(osp.join(base_path, str(index), f'rgb_{camera_index}.png'))
    rgb = np.asarray(rgb)[..., :3][..., ::-1]
    depth = np.load(osp.join(base_path, str(index), f'depth_{camera_index}.npy'))
    return rgb, depth / 1000


def load_intrinsics_and_extrinsics(base_path, camera_index=0):
    intrinsics = np.load(osp.join(base_path, f'intrinsics_{camera_index}.npy'))
    extrinsics = np.load(osp.join(base_path, f'extrinsics_{camera_index}.pkl'), allow_pickle=True)
    return intrinsics, extrinsics


def main():
    base_dir = '/Users/jiayuanm/Downloads/double_camera_hook/scooper'

    output = {'cameras': []}
    for j in range(2):
        i = 1  # Only read the first frame
        rgb, depth = load_rgbd(base_dir, i, j)
        intrinsics, extrinsics = load_intrinsics_and_extrinsics(base_dir, j)

        depth = (1000 * depth).astype(np.uint16)

        this_camera = {
            'rgb': cv2.imencode('.png', rgb)[1],
            'depth': cv2.imencode('.png', depth)[1],
            'intrinsics': intrinsics,
            'extrinsics': extrinsics
        }
        output['cameras'].append(this_camera)

    io.dump('./realsense-hook-test.pkl', output)


if __name__ == '__main__':
    main()

