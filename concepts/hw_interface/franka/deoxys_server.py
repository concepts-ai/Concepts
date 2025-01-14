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

import time
import atexit
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from threading import Barrier, Thread, Event
from copy import deepcopy

import cv2
import numpy as np

from jacinle.comm.service import Service, SocketClient
from jacinle.logging import get_logger

deoxys_available = True
try:
    from deoxys.franka_interface import FrankaInterface
    from deoxys.experimental.motion_utils import reset_joints_to_v1
    from deoxys.experimental.motion_utils import reset_joints_to_v2, follow_joint_traj, follow_ee_traj, open_gripper, close_gripper
    from deoxys.utils.config_utils import verify_controller_config, get_default_controller_config
    from deoxys.utils.yaml_config import YamlConfig
except ImportError:
    deoxys_available = False
    FrankaInterface = object

from concepts.vision.franka_system_calibration.calibration import get_mounted_camera_pose_from_qpos
from concepts.hw_interface.realsense.device_f import CaptureRS

__all__ = ['DeoxysService', 'DeoxysServiceVisualizerInterface', 'MoveQposTask', 'MoveQposTrajectoryTask', 'MultiRobotTaskBuilder', 'DeoxysClient']

logger = get_logger(__file__)


class DeoxysService(Service):
    DEFAULT_PORTS = (12347, 12348)

    ee_to_camera_pos = [0.036499, -0.034889, 0.0594]
    ee_to_camera_quat = [0.00252743, 0.0065769, 0.70345566, 0.71070423]

    def __init__(
        self,
        robots: Dict[int, FrankaInterface], robot_configs: Optional[Dict[int, np.ndarray]] = None,
        cameras: Optional[Dict[str, CaptureRS]] = None, camera_configs: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        camera_to_robot_mapping: Optional[Dict[str, int]] = None,
        mock: bool = False
    ):
        if not deoxys_available:
            raise ImportError('Deoxys is not available. Please install it first.')

        self.robots = robots
        self.robot_configs = robot_configs if robot_configs is not None else dict()
        self.cameras = cameras if cameras is not None else dict()
        self.camera_configs = camera_configs if camera_configs is not None else dict()
        self.camera_to_robot_mapping = camera_to_robot_mapping if camera_to_robot_mapping is not None else dict()
        self.visualizer = None
        self.mock = mock

        if True:
            logger.critical('Starting DeoxysService with the following configurations:')
            logger.info(f'Available robots: {list(self.robots.keys())}')
            logger.info(f'Available cameras: {list(self.cameras.keys())}')
            logger.info(f'Camera to robot mapping: {self.camera_to_robot_mapping}')
            logger.info(f'Robot configs: {self.robot_configs}')
            logger.info(f'Camera configs: {self.camera_configs}')

        super().__init__(configs=dict(), spec={'available_robots': list(self.robots.keys()), 'available_cameras': list(self.cameras.keys())})

    @classmethod
    def from_setup_name(cls, name: str, robot_configs: Optional[Dict[str, np.ndarray]], camera_configs: Optional[Dict[str, Dict[str, np.ndarray]]] = None, mock: bool = False):
        from concepts.hw_interface.franka.deoxys_interfaces import get_franka_interface_dict, get_realsense_capture_dict, get_setup_config
        from concepts.hw_interface.franka.deoxys_interfaces import get_robot_config_content_by_index

        setup_config = get_setup_config(name)
        robots = get_franka_interface_dict(setup_config['robots'], wait_for_state=True, auto_close=True)
        cameras = get_realsense_capture_dict(setup_config['cameras'], auto_close=True, skip_frames=30)

        camera_to_robot_mapping = dict()
        for robot_index in robots:
            robot_config = get_robot_config_content_by_index(robot_index)
            if 'GRIPPER_CAMERA_NAME' in robot_config['ROBOT']:
                camera_to_robot_mapping[robot_config['ROBOT']['GRIPPER_CAMERA_NAME']] = robot_index

        return cls(robots, robot_configs, cameras, camera_configs, camera_to_robot_mapping, mock)

    def start_visualizer(self):
        self.visualizer = DeoxysServiceVisualizerInterface(self)
        self.visualizer.start()

        import time; time.sleep(1.0)

        self.get_captures()

    def get_first_robot_index(self):
        return next(iter(self.robots))

    def get_robot_base_poses(self):
        # Return the configs['base_pose']
        return {robot_index: config['base_pose'] for robot_index, config in self.robot_configs.items()}

    def get_camera_to_robot_mapping(self):
        return self.camera_to_robot_mapping

    def get_camera_configs(self):
        camera_configs = dict()
        for k, cam in self.cameras.items():
            if k in self.camera_to_robot_mapping and self.camera_to_robot_mapping[k] in self.robots:
                robot = self.robots[self.camera_to_robot_mapping[k]]
                robot_base_pose = self.robot_configs[self.camera_to_robot_mapping[k]]['base_pose']
                extrinsics = robot_base_pose @ get_mounted_camera_pose_from_qpos(robot.last_q, self.ee_to_camera_pos, self.ee_to_camera_quat)
            elif k in self.camera_configs:
                extrinsics = self.camera_configs[k]['extrinsics']
            else:
                extrinsics = np.eye(4)
            camera_configs[k] = {
                'intrinsics': cam.intrinsics[0],
                'extrinsics': extrinsics
            }
        return camera_configs

    def get_captures(self, robot_move_to: Optional[Dict[int, np.ndarray]] = None):
        current_qpos = {robot_index: robot.last_q.copy() for robot_index, robot in self.robots.items()}
        if robot_move_to is not None:
            for robot_index, qpos in robot_move_to.items():
                self.single_move_qpos(robot_index, qpos)

        camera_configs = self.get_camera_configs()
        rv = dict()
        for k, cam in self.cameras.items():
            rgb, depth = cam.capture()
            rv[k] = {
                'color': rgb,
                'depth': depth,
                'intrinsics': camera_configs[k]['intrinsics'],
                'extrinsics': camera_configs[k]['extrinsics']
            }

        if self.visualizer is not None:
            self.visualizer.update_captures(rv)

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
        if cfg is None:
            cfg = get_default_controller_config('JOINT_IMPEDANCE')
        cfg = deepcopy(cfg)
        cfg = verify_controller_config(cfg, use_default=True)
        if residual_tau_translation is not None:
            cfg.enable_residual_tau = True
            cfg.residual_tau_translation_vec = residual_tau_translation
        follow_joint_traj(self.robots[robot_index], q, num_addition_steps=num_addition_steps, controller_cfg=cfg, gripper_open=gripper_open, gripper_close=gripper_close)

    def single_move_ee_trajectory(self, robot_index, ee_traj: List[Tuple[np.ndarray, np.ndarray]], *, cfg: Optional[Dict] = None, compliance_traj: Optional[List[np.ndarray]] = None, gripper_open: Optional[bool] = None, gripper_close: Optional[bool] = None):
        if self.mock:
            return print(f'DeoxysService::follow_ee_traj(robot_index={robot_index!r}, ee_traj={ee_traj!r}, controller_cfg={cfg!r})')
        follow_ee_traj(self.robots[robot_index], ee_traj, controller_cfg=cfg, compliance_traj=compliance_traj, gripper_open=gripper_open, gripper_close=gripper_close)

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


class DeoxysServiceVisualizerInterface(object):
    """Create a matplotlib visualizer for joint angles and end-effector poses."""
    def __init__(self, service: DeoxysService):
        self.service = service
        self.stop_event = Event()
        self.visualizer = None

        def atexit_handler():
            self.stop_event.set()

        atexit.register(atexit_handler)

    def start(self):
        thread = Thread(target=self.mainloop, daemon=True)
        thread.start()

    def update_captures(self, captures):
        print('Updating captures...')
        for name, data in captures.items():
            self.visualizer.update_queue(f'Camera_{name} Color', time.time(), data['color'], tab='Camera')
            depth = data['depth']
            depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
            self.visualizer.update_queue(f'Camera_{name} Depth', time.time(), depth_vis[..., ::-1], tab='Camera')

    def mainloop(self):
        # Create a subplot of 7 rows x N columns (N = number of robots)
        # For each robot, plot the joint angles
        from concepts.hw_interface.robot_state_visualizer.visualizer import RobotStateVisualizer
        self.visualizer = visualizer = RobotStateVisualizer('Franka System')
        for camera_index in self.service.cameras:
            visualizer.register_queue('Camera', f'Camera_{camera_index} Color', 'image', 1, group='Camera', width_in_group=6)
            visualizer.register_queue('Camera', f'Camera_{camera_index} Depth', 'image', 1, group='Camera', width_in_group=6)

        for robot_index in self.service.robots:
            for joint_index in range(1, 8):
                group_index = (joint_index - 1) // 4
                visualizer.register_queue(f'Robot{robot_index}', f'R{robot_index} Joint{joint_index}', 'float', 500, group=f'R{robot_index}Group{group_index}', width_in_group=3)
            for EE_axis in 'XYZ':
                visualizer.register_queue(f'Robot{robot_index}', f'R{robot_index} EE_{EE_axis}', 'float', 500, group=f'R{robot_index}EE_Pos', width_in_group=4)
            for EE_axis in 'XYZ':
                visualizer.register_queue(f'Robot{robot_index}', f'R{robot_index} EEForce_{EE_axis}', 'float', 500, group=f'R{robot_index}EE_Force', width_in_group=4)
        visualizer.start()

        tau_ext_initial = dict()
        for i, (robot_index, robot) in enumerate(self.service.robots.items()):
            tau_ext_initial[robot_index] = np.array(robot._state_buffer[-1].tau_J, copy=True) - np.array(robot._state_buffer[-1].generalized_gravity, copy=True)

        from concepts.simulator.pybullet.client import BulletClient
        from concepts.simulator.pybullet.components.panda.panda_robot import PandaRobot
        bc = BulletClient(is_gui=False)
        from concepts.benchmark.manip_tabletop.pybullet_tabletop_base.pybullet_tabletop import TableTopEnv
        env = TableTopEnv(bc)
        env.add_robot('panda')

        def compute_jacobian(robot, q):
            r: PandaRobot = env.robots[0]
            return r.get_jacobian(q, 11)

        def compute_ee_pos(robot, q):
            r: PandaRobot = env.robots[0]
            r.set_qpos(q)
            return r.get_ee_pose(fk=True)

        def update():
            if self.stop_event.is_set():
                return False

            for i, (robot_index, robot) in enumerate(self.service.robots.items()):
                qpos = np.array(robot._state_buffer[-1].q, copy=True)
                measured = np.array(robot._state_buffer[-1].tau_J, copy=True)
                gravity = np.array(robot._state_buffer[-1].generalized_gravity, copy=True)
                tau_ext = measured - gravity - tau_ext_initial[robot_index]

                jac_franka = np.array(robot._state_buffer[-1].jacobian_T_EE, copy=True).reshape(7, 6).T
                jac = compute_jacobian(robot, robot.last_q)

                cartesian_torque = np.linalg.pinv(jac.T) @ tau_ext
                cartesian_torque = cartesian_torque

                ee_pos, ee_quat = compute_ee_pos(robot, robot.last_q)

                time_f = time.time()
                for i in range(7):
                    visualizer.update_queue(f'R{robot_index} Joint{i+1}', time_f, qpos[i])

                visualizer.update_queue(f'R{robot_index} EE_X', time_f, ee_pos[0])
                visualizer.update_queue(f'R{robot_index} EE_Y', time_f, ee_pos[1])
                visualizer.update_queue(f'R{robot_index} EE_Z', time_f, ee_pos[2])

                cartesian_force = cartesian_torque[:3]
                from concepts.math.rotationlib_xyzw import quat2mat
                cartesian_force_in_world = np.linalg.inv(quat2mat(ee_quat)) @ cartesian_force

                visualizer.update_queue(f'R{robot_index} EEForce_X', time_f, cartesian_force_in_world[0])
                visualizer.update_queue(f'R{robot_index} EEForce_Y', time_f, cartesian_force_in_world[1])
                visualizer.update_queue(f'R{robot_index} EEForce_Z', time_f, cartesian_force_in_world[2])

            time.sleep(0.1)
            return True

        while True:
            if not update():
                break


class DeoxysClient(object):
    def __init__(self, server: str, ports: Optional[Tuple[int, int]] = None):
        if ports is None:
            ports = DeoxysService.DEFAULT_PORTS
        connection = (f'tcp://{server}:{ports[0]}', f'tcp://{server}:{ports[1]}')
        self.client = SocketClient('concepts::deoxys_interface', connection)
        self.client.initialize(auto_close=True)
        self.default_robot_index = self.client.call('get_first_robot_index')

    def get_default_robot_index(self):
        return self.default_robot_index

    def set_default_robot_index(self, index):
        self.default_robot_index = index

    def get_robot_base_poses(self):
        return self.client.call('get_robot_base_poses')

    def get_camera_to_robot_mapping(self):
        return self.client.call('get_camera_to_robot_mapping')

    def get_camera_configs(self):
        return self.client.call('get_camera_configs')

    def get_captures(self, robot_move_to: Optional[Dict[int, np.ndarray]] = None):
        return self.client.call('get_captures', robot_move_to)

    def get_all_qpos(self):
        return self.client.call('get_all_qpos')

    def get_all_qvel(self):
        return self.client.call('get_all_qvel')

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

    def single_move_ee_trajectory(self, robot_index, ee_traj: List[Tuple[np.ndarray, np.ndarray]], *, cfg: Optional[Dict] = None, compliance_traj: Optional[List[np.ndarray]] = None, gripper_open: Optional[bool] = None, gripper_close: Optional[bool] = None):
        self.client.call('single_move_ee_trajectory', robot_index, ee_traj, cfg=cfg, compliance_traj=compliance_traj, gripper_open=gripper_open, gripper_close=gripper_close)

    def get_qpos(self):
        return self.client.call('get_all_qpos')[self.default_robot_index]

    def get_qvel(self):
        return self.client.call('get_all_qvel')[self.default_robot_index]

    def open_gripper(self, steps: int = 100):
        self.client.call('single_open_gripper', self.default_robot_index, steps)

    def close_gripper(self, steps: int = 100):
        self.client.call('single_close_gripper', self.default_robot_index, steps)

    def move_home(self, gripper_open: Optional[bool] = None, gripper_close: Optional[bool] = None):
        self.client.call('single_move_home', self.default_robot_index, gripper_open=gripper_open, gripper_close=gripper_close)

    def move_qpos(self, q: np.ndarray, cfg: Optional[Dict] = None, gripper_open: Optional[bool] = None, gripper_close: Optional[bool] = None):
        self.client.call('single_move_qpos', self.default_robot_index, q, cfg, gripper_open=gripper_open, gripper_close=gripper_close)

    def move_qpos_trajectory(self, q: List[np.ndarray], cfg: Optional[Dict] = None, num_addition_steps: int = 0, gripper_open: Optional[bool] = None, gripper_close: Optional[bool] = None, residual_tau_translation: Optional[np.ndarray] = None):
        self.client.call('single_move_qpos_trajectory', self.default_robot_index, q, cfg, num_addition_steps=num_addition_steps, gripper_open=gripper_open, gripper_close=gripper_close, residual_tau_translation=residual_tau_translation)

    def move_ee_trajectory(self, ee_traj: List[Tuple[np.ndarray, np.ndarray]], *, cfg: Optional[Dict] = None, compliance_traj: Optional[List[np.ndarray]] = None, gripper_open: Optional[bool] = None, gripper_close: Optional[bool] = None):
        self.client.call('single_move_ee_trajectory', self.default_robot_index, ee_traj, cfg=cfg, compliance_traj=compliance_traj, gripper_open=gripper_open, gripper_close=gripper_close)

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

