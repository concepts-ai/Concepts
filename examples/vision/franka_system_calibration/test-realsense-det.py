#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test-realsense-det.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/01/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.
from ossaudiodev import control_names

import cv2
import time
import atexit
import numpy as np
import jacinle
from typing import Tuple

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.experimental.motion_utils import reset_joints_to

from concepts.hw_interface.realsense.device import RealSenseDevice
from concepts.math.rotationlib_xyzw import quat2mat, mat2quat
from concepts.math.frame_utils_xyzw import compose_transformation
from concepts.vision.franka_system_calibration.ar_detection import get_ar_tag_detections, get_ar_tag_poses_from_camera_pose
from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.pybullet.components.panda.panda_robot import PandaRobot


def get_camera_pose(robot, qpos):
    ee_pos, ee_quat = robot.fk(qpos, 8)

    ee_to_camera_pos = [0.036499, - 0.034889, 0.0594]
    ee_to_camera_quat = [0.00252743, 0.0065769, 0.70345566, 0.71070423]

    pos, quat = compose_transformation(ee_pos, ee_quat, ee_to_camera_pos, ee_to_camera_quat)

    camera_pose = np.eye(4)
    camera_pose[:3, :3] = quat2mat(quat)
    camera_pose[:3, 3] = pos
    return camera_pose


def get_pos_quat_from_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pos = matrix[:3, 3]
    quat = mat2quat(matrix[:3, :3])
    return pos, quat


def move_to_target_pose(robot_interface, robot, pos, quat):
    qpos = robot.ikfast(pos, quat)
    print(qpos)
    reset_joints_to(robot_interface, qpos, gripper_open=True)
    reset_joints_to(robot_interface, qpos, gripper_open=False)


def get_ar_tag_poses(robot_index = 1):
    if robot_index == 1:
        franka_interface = FrankaInterface(config_root + "/charmander.yml", use_visualizer=False)
    elif robot_index == 2:
        franka_interface = FrankaInterface(config_root + "/charmander2.yml", use_visualizer=False)
    else:
        raise ValueError('Invalid robot index.')

    def at_exit_franka():
        franka_interface.close()
        print('Stopped Franka interface.')
    atexit.register(at_exit_franka)

    bullet_client = BulletClient(is_gui=False)
    panda_robot = PandaRobot(bullet_client)

    device = RealSenseDevice.from_serial_number(franka_interface.gripper_camera)
    device.start_pipeline()

    def at_exit_rs():
        device.stop_pipeline()
    atexit.register(at_exit_rs)

    while franka_interface.last_q is None:
        print('Waiting for robot state...')
        import time; time.sleep(0.1)

    color, depth = device.capture_images()

    dets = get_ar_tag_detections(color)
    print(dets)

    qpos = franka_interface.last_q
    print(qpos)

    cam_pose = get_camera_pose(panda_robot, qpos)
    print(cam_pose)

    poses = get_ar_tag_poses_from_camera_pose(dets, camera_intrinsics=device.color_intrinsics, camera_pose=cam_pose)
    print(poses)

    panda_robot.set_qpos(qpos)
    print(panda_robot.get_qpos())
    print(panda_robot.get_ee_pose())

    data = {
        'index': robot_index,
        'color': color,
        'depth': depth,
        'dets': dets,
        'qpos': qpos,
        'poses': poses,
        'intrinsics': device.color_intrinsics,
        'extrinsics': cam_pose
    }

    data_filename = f'./calibration-{robot_index}.pkl'
    jacinle.dump(data_filename, data)
    print(f'Saved data to {data_filename}')

    bullet_client.disconnect()
    return franka_interface, device, poses


def get_capture_scene(robot_index = 1):
    if robot_index == 1:
        franka_interface = FrankaInterface(config_root + "/charmander.yml", use_visualizer=False)
    elif robot_index == 2:
        franka_interface = FrankaInterface(config_root + "/charmander2.yml", use_visualizer=False)
    else:
        raise ValueError('Invalid robot index.')

    def at_exit_franka():
        franka_interface.close()
        print('Stopped Franka interface.')

    atexit.register(at_exit_franka)

    device = RealSenseDevice.from_serial_number(franka_interface.gripper_camera)
    device.start_pipeline()

    def at_exit_rs():
        device.stop_pipeline()
    atexit.register(at_exit_rs)

    for i in range(5):
        print(f'Waiting for camera to stabilize... {i+1}/5')
        device.capture_images()
        time.sleep(0.1)

    color, depth = device.capture_images()
    data = {
        'index': robot_index,
        'color': color,
        'depth': depth,
        'intrinsics': device.color_intrinsics,
    }

    data_filename = f'./scene-{robot_index}.pkl'
    jacinle.dump(data_filename, data)
    print(f'Saved data to {data_filename}')


def main():
    fr1, rs1, poses1 = get_ar_tag_poses(1)
    fr2, rs2, poses2 = get_ar_tag_poses(2)

    bullet_client = BulletClient(is_gui=False)
    panda_robot = PandaRobot(bullet_client)

    print(poses1)
    print(poses2)

    # pos, quat = get_pos_quat_from_matrix(poses1[0])
    # pos = pos + np.array([0, 0, 0.02])
    # move_to_target_pose(fr1, panda_robot, pos, (-0.707, 0.707, 0, 0))

    # pos, quat = get_pos_quat_from_matrix(poses2[0])
    # pos = pos + np.array([0, 0, 0.04])
    # move_to_target_pose(fr2, panda_robot, pos, (0, 1, 0, 0))


def main_capture_scene():
    get_capture_scene(1)
    get_capture_scene(2)


if __name__ == '__main__':
    # main()
    main_capture_scene()
