#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : deoxys_server.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/10/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Control interface for the Panda robot using the Deoxys library."""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from threading import Barrier, Thread

from jacinle.comm.service import Service, SocketClient

deoxys_available = True
try:
    from deoxys.franka_interface import FrankaInterface
    from deoxys.experimental.motion_utils import reset_joints_to_v2, follow_joint_traj, follow_ee_traj, open_gripper, close_gripper
    from deoxys.utils.config_utils import verify_controller_config
    from deoxys.utils.yaml_config import YamlConfig
except ImportError:
    deoxys_available = False
    FrankaInterface = object

from concepts.vision.franka_system_calibration.calibration import get_mounted_camera_pose_from_qpos
from concepts.hw_interface.realsense.device_f import CaptureRS

__all__ = ['DeoxysService', 'MoveQposTask', 'MoveQposTrajectoryTask', 'MultiRobotTaskBuilder']


class DeoxysService(Service):
    DEFAULT_PORTS = (12347, 12348)

    def __init__(self, robots: Dict[int, FrankaInterface], camera_configs: Optional[Dict[str, Dict[str, np.ndarray]]] = None, cameras: Optional[Dict[str, CaptureRS]] = None, mock: bool = False):
        if not deoxys_available:
            raise ImportError('Deoxys is not available. Please install it first.')

        self.robots = robots
        self.camera_configs = camera_configs if camera_configs is not None else {}
        self.cameras = cameras if cameras is not None else {}
        self.mock = mock

        super().__init__(configs=dict(), spec={'available_robots': list(self.robots.keys()), 'available_cameras': list(self.cameras.keys())})

    @classmethod
    def from_setup_name(cls, name: str, camera_configs: Optional[Dict[str, Dict[str, np.ndarray]]] = None, mock: bool = False):
        from concepts.hw_interface.franka.deoxys_interfaces import get_franka_interface_dict, get_realsense_capture_dict, get_setup_config
        setup_config = get_setup_config(name)
        robots = get_franka_interface_dict(setup_config['robots'], wait_for_state=True, auto_close=True)
        cameras = get_realsense_capture_dict(setup_config['cameras'], auto_close=True, skip_frames=30)
        return cls(robots, camera_configs, cameras, mock)

    def get_captures(self, robot_move_to: Optional[Dict[int, np.ndarray]] = None):
        current_qpos = {robot_index: robot.last_q.copy() for robot_index, robot in self.robots.items()}
        if robot_move_to is not None:
            for robot_index, qpos in robot_move_to.items():
                self.single_move_qpos(robot_index, qpos)

        rv = dict()
        for k, cam in self.cameras.items():
            if k == 'robot1_hand':
                ee_to_camera_pos = [0.036499, -0.034889, 0.0594]
                ee_to_camera_quat = [0.00252743, 0.0065769, 0.70345566, 0.71070423]
                extrinsics = get_mounted_camera_pose_from_qpos(self.robots[1].last_q, ee_to_camera_pos, ee_to_camera_quat)
            elif k in self.camera_configs:
                extrinsics = self.camera_configs[k]['extrinsics']
            else:
                extrinsics = np.eye(4)
            rgb, depth = cam.capture()
            rv[k] = {
                'color': rgb,
                'depth': depth,
                'intrinsics': cam.intrinsics[0],
                'extrinsics': extrinsics
            }

        if robot_move_to is not None:
            for robot_index, qpos in current_qpos.items():
                self.single_move_qpos(robot_index, qpos)
        return rv

    def get_all_qpos(self):
        return {robot_index: robot.last_q.copy() for robot_index, robot in self.robots.items()}

    def get_all_qvel(self):
        return {robot_index: robot.last_dq.copy() for robot_index, robot in self.robots.items()}

    def get_full_observation(self):
        return {
            'qpos': self.get_all_qpos(),
            'qvel': self.get_all_qvel(),
            'captures': self.get_captures()
        }

    def single_open_gripper(self, robot_index, steps: int = 100):
        if self.mock:
            return print(f'DeoxysService::open_gripper(robot_index={robot_index!r}, steps={steps!r})')
        open_gripper(self.robots[robot_index], steps)

    def single_close_gripper(self, robot_index, steps: int = 100):
        if self.mock:
            return print(f'DeoxysService::close_gripper(robot_index={robot_index!r}, steps={steps!r})')
        close_gripper(self.robots[robot_index], steps)

    def single_move_home(self, robot_index: int, gripper_open: Optional[bool] = None, gripper_close: Optional[bool] = None):
        reset_joints_to_v2(self.robots[robot_index], self.robots[robot_index].init_q, gripper_open=gripper_open, gripper_close=gripper_close, gripper_default='open')

    def single_move_qpos(self, robot_index: int, q: np.ndarray, cfg: Optional[Dict] = None, gripper_open: Optional[bool] = None, gripper_close: Optional[bool] = None):
        q = np.asarray(q)
        if q.shape != (7,):
            raise ValueError('Invalid joint position.')

        if self.mock:
            return print(f'DeoxysService::reset_joints_to_v2(robot_index={robot_index!r}, q={q!r}, controller_cfg={cfg!r})')
        reset_joints_to_v2(self.robots[robot_index], q, controller_cfg=cfg, gripper_open=gripper_open, gripper_close=gripper_close)

    def single_move_qpos_trajectory(self, robot_index: int, q: List[np.ndarray], cfg: Optional[Dict] = None, num_addition_steps: int = 0, gripper_open: Optional[bool] = None, gripper_close: Optional[bool] = None, residual_tau_translation: Optional[np.ndarray] = None):
        if self.mock:
            return print(f'DeoxysService::follow_joint_traj(robot_index={robot_index!r}, q={q!r}, num_addition_steps={num_addition_steps}, controller_cfg={cfg!r})')
        cfg = verify_controller_config(cfg, use_default=True)
        if residual_tau_translation is not None:
            cfg.enable_residual_tau = True
            cfg.residual_tau_translation_vec = residual_tau_translation
        follow_joint_traj(self.robots[robot_index], q, num_addition_steps=num_addition_steps, controller_cfg=cfg, gripper_open=gripper_open, gripper_close=gripper_close)

    def single_move_ee_trajectory(self, robot_index, ee_traj: List[Tuple[np.ndarray, np.ndarray]], compliance_traj: Optional[List[np.ndarray]] = None, cfg: Optional[Dict] = None, gripper_open: Optional[bool] = None, gripper_close: Optional[bool] = None):
        if self.mock:
            return print(f'DeoxysService::follow_ee_traj(robot_index={robot_index!r}, ee_traj={ee_traj!r}, controller_cfg={cfg!r})')
        follow_ee_traj(self.robots[robot_index], ee_traj, compliance_traj=compliance_traj, controller_cfg=cfg, gripper_open=gripper_open, gripper_close=gripper_close)

    def run_multi_robot(self, task: List[Tuple[str, object]], activated_robots: Optional[List[int]] = None):
        builder = MultiRobotTaskBuilder(task)
        builder.run(self, activated_robots)

    def call(self, name, *args, **kwargs):
        return getattr(self, name)(*args, **kwargs)

    def serve_socket(self, name=None, tcp_port=None):
        if name is None:
            name = 'concepts::deoxys_interface'
        if tcp_port is None:
            tcp_port = DeoxysService.DEFAULT_PORTS
        super().serve_socket(name, tcp_port)


class DeoxysClient(object):
    def __init__(self, server: str, ports: Optional[Tuple[int, int]] = None):
        if ports is None:
            ports = DeoxysService.DEFAULT_PORTS
        connection = (f'tcp://{server}:{ports[0]}', f'tcp://{server}:{ports[1]}')
        self.client = SocketClient('concepts::deoxys_interface', connection)
        self.client.initialize(auto_close=True)

    def get_captures(self, robot_move_to: Optional[Dict[int, np.ndarray]] = None):
        return self.client.call('get_captures', robot_move_to)

    def get_all_qpos(self):
        return self.client.call('get_all_qpos')

    def get_all_qvel(self):
        return self.client.call('get_all_qvel')

    DEFAULT_ROBOT_INDEX = 1

    def single_open_gripper(self, robot_index, steps: int = 100):
        self.client.call('single_open_gripper', robot_index, steps)

    def single_close_gripper(self, robot_index, steps: int = 100):
        self.client.call('single_close_gripper', robot_index, steps)

    def single_move_home(self, robot_index: int, gripper_open: Optional[bool] = None, gripper_close: Optional[bool] = None):
        self.client.call('single_move_home', robot_index, gripper_open=gripper_open, gripper_close=gripper_close)

    def single_move_qpos(self, robot_index: int, q: np.ndarray, cfg: Optional[Dict] = None, gripper_open: Optional[bool] = None, gripper_close: Optional[bool] = None):
        self.client.call('single_move_qpos', robot_index, q, cfg, gripper_open=gripper_open, gripper_close=gripper_close)

    def single_move_qpos_trajectory(self, robot_index: int, q: List[np.ndarray], cfg: Optional[Dict] = None, num_addition_steps: int = 0, gripper_open: Optional[bool] = None, gripper_close: Optional[bool] = None, residual_tau_translation: Optional[np.ndarray] = None):
        self.client.call('single_move_qpos_trajectory', robot_index, q, cfg, num_addition_steps=num_addition_steps, gripper_open=gripper_open, gripper_close=gripper_close, residual_tau_translation=residual_tau_translation)

    def single_move_ee_trajectory(self, robot_index, ee_traj: List[Tuple[np.ndarray, np.ndarray]], cfg: Optional[Dict] = None, compliance_traj: Optional[List[np.ndarray]] = None, gripper_open: Optional[bool] = None, gripper_close: Optional[bool] = None):
        self.client.call('single_move_ee_trajectory', robot_index, ee_traj, cfg, compliance_traj=compliance_traj, gripper_open=gripper_open, gripper_close=gripper_close)

    def get_qpos(self):
        return self.client.call('get_all_qpos')[type(self).DEFAULT_ROBOT_INDEX]

    def get_qvel(self):
        return self.client.call('get_all_qvel')[type(self).DEFAULT_ROBOT_INDEX]

    def open_gripper(self, steps: int = 100):
        self.client.call('single_open_gripper', type(self).DEFAULT_ROBOT_INDEX, steps)

    def close_gripper(self, steps: int = 100):
        self.client.call('single_close_gripper', type(self).DEFAULT_ROBOT_INDEX, steps)

    def move_home(self, gripper_open: Optional[bool] = None, gripper_close: Optional[bool] = None):
        self.client.call('single_move_home', type(self).DEFAULT_ROBOT_INDEX, gripper_open=gripper_open, gripper_close=gripper_close)

    def move_qpos(self, q: np.ndarray, cfg: Optional[Dict] = None, gripper_open: Optional[bool] = None, gripper_close: Optional[bool] = None):
        self.client.call('single_move_qpos', type(self).DEFAULT_ROBOT_INDEX, q, cfg, gripper_open=gripper_open, gripper_close=gripper_close)

    def move_qpos_trajectory(self, q: List[np.ndarray], cfg: Optional[Dict] = None, num_addition_steps: int = 0, gripper_open: Optional[bool] = None, gripper_close: Optional[bool] = None, residual_tau_translation: Optional[np.ndarray] = None):
        self.client.call('single_move_qpos_trajectory', type(self).DEFAULT_ROBOT_INDEX, q, cfg, num_addition_steps=num_addition_steps, gripper_open=gripper_open, gripper_close=gripper_close, residual_tau_translation=residual_tau_translation)

    def move_ee_trajectory(self, ee_traj: List[Tuple[np.ndarray, np.ndarray]], cfg: Optional[Dict] = None, num_additional_steps: int = 0, compliance_traj: Optional[List[np.ndarray]] = None, gripper_open: Optional[bool] = None, gripper_close: Optional[bool] = None):
        if num_additional_steps > 0:
            raise NotImplementedError('additional_steps is not supported in DeoxysClient::move_ee_trajectory.')
        self.client.call('single_move_ee_trajectory', type(self).DEFAULT_ROBOT_INDEX, ee_traj, cfg, compliance_traj=compliance_traj, gripper_open=gripper_open, gripper_close=gripper_close)

    def run_multi_robot(self, task: List[Tuple[str, object]], activated_robots: Optional[List[int]] = None):
        self.client.call('run_multi_robot', task, activated_robots)


@dataclass
class MoveQposTask(object):
    qpos_dict: Dict[int, np.ndarray]
    cfg_dict: Dict[int, Dict]


@dataclass
class MoveQposTrajectoryTask(object):
    qpos_traj_dict: Dict[int, List[np.ndarray]]
    cfg_dict: Dict[int, Dict]


@dataclass
class MoveGripperTask(object):
    open: bool
    steps: int


class MultiRobotTaskBuilder(object):
    def __init__(self, tasks: Optional[List[Tuple[str, object]]] = None):
        if tasks is None:
            tasks = list()
        self.tasks = tasks

    def get_tasks(self):
        # Note: not deep copy
        return self.tasks.copy()

    def move_qpos(self, tasks: Dict[int, np.ndarray], configs: Optional[Dict[int, Dict]] = None):
        if configs is None:
            configs = dict()
        self.tasks.append(('move_qpos', MoveQposTask(tasks, configs)))

    def move_qpos_trajectorry(self, tasks: Dict[int, List[np.ndarray]], configs: Optional[Dict[int, Dict]] = None):
        if configs is None:
            configs = dict()
        self.tasks.append(('move_qpos_trajectory', MoveQposTrajectoryTask(tasks, configs)))

    def open_gripper(self, steps: int = 100):
        self.tasks.append(('open_gripper', MoveGripperTask(True, steps)))

    def close_gripper(self, steps: int = 100):
        self.tasks.append(('close_gripper', MoveGripperTask(False, steps)))

    def thread_run(self, interface: DeoxysService, index: int, barriers: List):
        for i in range(len(self.tasks)):
            task, payload = self.tasks[i]
            if task == 'move_qpos':
                assert isinstance(payload, MoveQposTask)
                if index in payload.qpos_dict:
                    interface.single_move_qpos(index, payload.qpos_dict[index], payload.cfg_dict.get(index, None))
            elif task == 'move_qpos_trajectory':
                assert isinstance(payload, MoveQposTrajectoryTask)
                if index in payload.qpos_traj_dict:
                    interface.single_move_qpos_trajectory(index, payload.qpos_traj_dict[index], payload.cfg_dict.get(index, None))
            elif task == 'move_gripper':
                assert isinstance(payload, MoveGripperTask)
                if payload.open:
                    interface.single_open_gripper(index, payload.steps)
                else:
                    interface.single_close_gripper(index, payload.steps)
            else:
                raise ValueError(f'Invalid task: {task!r}')
            barriers[i].wait()

    def run(self, interface: DeoxysService, activated_robots: Optional[List[int]] = None):
        if activated_robots is None:
            activated_robots = list(interface.robots.keys())

        threads = list()
        barriers = [Barrier(len(activated_robots)) for _ in range(len(self.tasks))]
        for robot_index in activated_robots:
            thread = Thread(target=self.thread_run, args=(interface, robot_index, barriers))
            threads.append(thread)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def run_remote(self, client: DeoxysClient, activated_robots: Optional[List[int]] = None):
        client.run_multi_robot(self.get_tasks(), activated_robots)

