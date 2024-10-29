#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : remote_client.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/08/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import atexit
import itertools
from typing import Optional, Union, List

import numpy as np
import cv2

from concepts.hw_interface.franka.server import FrankaServiceClient
from concepts.math.rotationlib_xyzw import quat_mul, quat_conjugate, rotate_vector


def print_franka_cli_welcome():
    # Draw an ASCII art of a panda
    panda_string = """
            ██████                        ██████    
          ██████████  ████████████████  ██████████  
        ██████████████                ██████████████
        ████████                            ████████
        ██████                                ██████
          ██                                    ██  
          ██                                    ██  
        ██        ██████            ██████        ██
        ██      ██████████        ██████████      ██
        ██    ████████  ██        ██  ████████    ██
        ██    ████████  ██        ██  ████████    ██
        ██    ██████████            ██████████    ██
        ██      ██████      ████      ██████      ██
          ██                ████                ██  
          ████████  ▒▒▒▒▒▒        ▒▒▒▒▒▒  ████████  
        ████████████▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒▒████████████
        ██████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██████████████
        ██████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██████████████
        ██████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██████████████
          ████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒████████████  
            ████████  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  ████████    
                        ▒▒▒▒▒▒▒▒▒▒▒▒                
                          ▒▒▒▒▒▒▒▒                  
                            ▒▒▒▒                    

                 Panda Robot Interactive CLI        
    """
    string_start = '\x1b[30m\x1b[107m'
    string_end = '\x1b[39m\x1b[49m'
    for s in panda_string.splitlines():
        s = s + ' ' * (52 - len(s))
        print(string_start + s + string_end)
    print()
    print('Welcome to the Franka interactive command line.')


class FrankaRemoteClient(object):
    def __init__(self, server: str, name='franka-client', port_pair=None, automatic_reset_errors: bool = True):
        self.service = FrankaServiceClient(server, name, port_pair, automatic_reset_errors)
        self.service.initialize()
        atexit.register(self._atexit)
        self.ikfast_wrapper = None

    def _atexit(self):
        if self.service is not None:
            self.service.finalize()
            self.service = None

    def __del__(self):
        self._atexit()

    def open_gripper(self, width: float = 0.2) -> None:
        self.service.open_gripper(width)

    def close_gripper(self) -> None:
        self.service.close_gripper()

    def grasp(self, width: float = 0.05, force: float = 40):
        self.service.grasp_gripper(width, force)

    def visualize_camera_rgb(self):
        rgb, _ = self.service.capture_image()
        cv2.imshow('Camera RGB', rgb)
        cv2.waitKey(0)

    def print_pose(self):
        pos, quat = self.get_ee_pose()
        print(f'Actual pose: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}')
        print(f'Actual quat: x={quat[0]:.4f}, y={quat[1]:.4f}, z={quat[2]:.4f}, w={quat[3]:.4f}')

    def get_ee_pose(self):
        return self.service.get_ee_pose()

    def get_qpos(self):
        return self.service.get_qpos()

    def move_home(self):
        self.service.move_home()

    def move_qpos(self, qpos: np.ndarray, timeout: float = 10) -> None:
        self.service.move_qpos(qpos, timeout)

    def move_pose(self, pos, quat=None, timeout: float = 10):
        if quat is None:
            quat = (0, 1, 0, 0)
        pos = np.array(pos)
        quat = np.array(quat)
        last_qpos = self.get_qpos()
        ik_solution = self.ikfast(pos, quat, last_qpos, error_on_fail=False)
        if ik_solution is None:
            print('Failed to find IK.')
            return False

        self.move_qpos(ik_solution, timeout)
        return True

    def move_qpos_trajectory(self, qpos_list, timeout: float = 10) -> None:
        current_qpos = self.get_qpos()
        qpos_list = [current_qpos] + qpos_list
        self.service.move_qpos_trajectory(qpos_list, timeout)

    def reset_errors(self) -> None:
        return self.service.reset_errors()

    def set_manual_control(self):
        self.service.reset_errors()
        self.service.set_cart_impedance_freemotion()
        self.service.reset_errors()

    def run_calibration_fix(self):
        """Helpful function for fixing an object at the current position. The robot will grasp, open, and move home."""
        self.grasp()
        # self.move_home()
        # self.reset_errors()

    def _init_ikfast(self):
        if self.ikfast_wrapper is None:
            from concepts.simulator.ikfast.ikfast_common import IKFastWrapperBase
            import concepts.simulator.ikfast.franka_panda.ikfast_panda_arm as ikfast_module
            self.ikfast_wrapper = IKFastWrapperBase(
                ikfast_module,
                joint_ids=[0, 1, 2, 3, 4, 5, 6], free_joint_ids=[6], use_xyzw=True,
                joints_lower=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175,-2.8973],
                joints_upper=[2.8963, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
            )

    def ikfast(
        self, pos: np.ndarray, quat: np.ndarray, last_qpos: np.ndarray,
        max_attempts: int = 1000, max_distance: float = float('inf'), error_on_fail: bool = True,
    ) -> Optional[Union[List[np.ndarray], np.ndarray]]:
        self._init_ikfast()

        pos_delta = [0, 0, 0.1]
        quat_delta = (0.0, 0.0, 0.9238795325108381, 0.38268343236617297)
        inner_quat = quat_mul(quat, quat_conjugate(quat_delta))
        inner_pos = np.array(pos) - rotate_vector(pos_delta, quat)

        try:
            ik_solution = list(itertools.islice(self.ikfast_wrapper.gen_ik(inner_pos, inner_quat, last_qpos=last_qpos, max_attempts=max_attempts, max_distance=max_distance, verbose=False), 1))[0]
        except IndexError:
            if error_on_fail:
                raise
            return None

        return ik_solution
