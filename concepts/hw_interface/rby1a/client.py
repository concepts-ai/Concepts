#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : client.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/16/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""UPC Client for the RBY1 robot."""

import io
import numpy as np
from typing import Optional, Union, Tuple, List, Dict

from jacinle.comm.service import SocketClient


class RBY1AInterfaceClient(object):
    DEFAULT_PORTS = (12345, 12346)
    DEFAULT_NAME = 'concepts::rby1_interface'

    def __init__(self, server: str, ports: Optional[Tuple[int, int]] = None, name: str = DEFAULT_NAME):
        if ports is None:
            ports = self.DEFAULT_PORTS

        connection = (f'tcp://{server}:{ports[0]}', f'tcp://{server}:{ports[1]}')
        self.client = SocketClient(name, connection)
        self.client.initialize(auto_close=True)
        self.reset_stream()

    def power_on(self):
        return self.client.call('power_on')

    def power_off(self):
        return self.client.call('power_off')

    def reset_stream(self):
        return self.client.call('reset_stream')

    def get_qpos(self) -> Dict[str, np.ndarray]:
        return self.client.call('get_qpos')

    def get_vector_qpos(self) -> np.ndarray:
        return self.client.call('get_vector_qpos')

    def get_captures(self) -> Dict[str, Dict[str, np.ndarray]]:
        return self.client.call('get_captures')

    def get_captures_compressed(self) -> Dict[str, Dict[str, np.ndarray]]:
        values = self.client.call('get_captures_compressed')
        for k, v in values.items():
            buf = io.BytesIO(v)
            values[k] = np.load(buf, allow_pickle=True)
        return values

    def move_zero(self, min_time: float = 10):
        return self.client.call('move_zero', min_time)

    def move_qpos(self, qpos: Dict[str, Union[np.ndarray, list]], timeout: float = 5, min_time: float = 1.666):
        return self.client.call('move_qpos', qpos, timeout, min_time)

    def move_qpos_if_not_close(self, qpos: Dict[str, Union[np.ndarray, list]], atol: float = 1e-3, timeout: float = 5, min_time: float = 1.666):
        current_qpos = self.get_qpos()
        if close_to(current_qpos, qpos, atol=atol):
            return
        self.move_qpos(qpos, timeout=timeout, min_time=min_time)

    def move_qpos_trajectory(self, trajectory: List[Dict[str, Union[np.ndarray, list]]], dt: float = 0.5, min_time_multiplier = 5, first_min_time: Optional[float] = None, command_timeout: float = 1.0):
        return self.client.call('move_qpos_trajectory', trajectory, dt, min_time_multiplier=min_time_multiplier, first_min_time=first_min_time, command_timeout=command_timeout)

    def move_gripper_qpos_trajectory(self, gripper_name: str, trajectory: Union[np.ndarray, list]):
        return self.client.call('move_gripper_qpos_trajectory', gripper_name, trajectory)

    def move_gripper_percent_trajectory(self, gripper_name: str, trajectory: Union[np.ndarray, list]):
        return self.client.call('move_gripper_percent_trajectory', gripper_name, trajectory)

    def move_gripper_percent(self, gripper_name: str, percent: float):
        return self.client.call('move_gripper_percent_trajectory', gripper_name, [percent])

    def move_gripper_open(self, gripper_name: str):
        return self.move_gripper_percent(gripper_name, 1.0)

    def move_gripper_close(self, gripper_name: str, remaining_percent: float = 0.0):
        return self.move_gripper_percent(gripper_name, remaining_percent)


def close_to(state, reference, atol=1e-3):
    for k, v in reference.items():
        if not np.allclose(state[k], v, atol=atol):
            return False
    return True

