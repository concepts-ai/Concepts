#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : server.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/16/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""UPC Server for the RBY1 robot."""

import io
import sys
import queue
import jacinle
import atexit
import time
import numpy as np
from typing import Optional, List, Dict
from threading import Thread, Event

import cv2
import rby1_sdk

from jacinle.comm.service import Service
from concepts.hw_interface.realsense.device_f import CaptureRS
from concepts.hw_interface.rby1a.qr_gripper_controller import RainbowGripperController
from concepts.hw_interface.rby1a.client import RBY1AInterfaceClient
from concepts.math.frame_utils_xyzw import frame_mul
from concepts.math.rotationlib_xyzw import mat2pos_quat_xyzw, pos_quat2mat_xyzw
from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.pybullet.components.rby1a.rby1a_robot import RBY1ARobot


class RBY1AUPCInterface(Service):
    DEFAULT_ROBOT_ADDRESS = "192.168.30.1:50051"
    DEFAULT_CONFIGS = {
        "joint_position_command.cutoff_frequency": "5",
        "default.acceleration_limit_scaling": "0.8",
    }
    DEFAULT_PRIORITY = 10
    RS_SERIAL_NUMBER = '231122072622'
    USE_STREAM_INTERFACE = True

    def __init__(self, connect: bool = True, init_camera: bool = True, init_grippers: bool = True, address: str = DEFAULT_ROBOT_ADDRESS, priority: int = DEFAULT_PRIORITY):
        super().__init__({
            'robot_address': address,
            'configs': self.DEFAULT_CONFIGS,
            'priority': priority
        })

        self._address = address
        self._priority = priority
        self._connected = False
        self._robot = None
        self._pb_robot = None
        self._gripper_controller = None
        self._camera = None

        self._current_controller = None
        self._current_stream = None

        T_headlink2_to_cameraoptical = np.array([
                [ 0.        ,  0.        ,  1.        ,  0.04756366],
                [-1.        ,  0.        ,  0.        ,  0.02982876],
                [ 0.        , -1.        ,  0.        ,  0.06316296],
                [ 0.        ,  0.        ,  0.        ,  1.        ]
        ])
        self._camera_headlink2_to_cameraoptical = T_headlink2_to_cameraoptical
        assert self.pb_robot is not None  # Initialize the robot model

        if connect:
            self.connect(init_camera=init_camera, init_grippers=init_grippers)

    @property
    def address(self) -> str:
        return self._address

    @property
    def robot(self) -> rby1_sdk.Robot_A:
        assert self._robot is not None, "Robot is not initialized."
        return self._robot

    @property
    def pb_robot(self) -> RBY1ARobot:
        if self._pb_robot is None:
            bclient = BulletClient(is_gui=False)
            self._pb_robot = RBY1ARobot(bclient)
        return self._pb_robot

    @property
    def has_gripper_controller(self) -> bool:
        return self._gripper_controller is not None

    @property
    def gripper_controller(self) -> RainbowGripperController:
        assert self._gripper_controller is not None, "Gripper controller is not initialized."
        return self._gripper_controller

    @property
    def has_camera(self) -> bool:
        return self._camera is not None

    @property
    def camera(self) -> CaptureRS:
        assert self._camera is not None, "Camera is not initialized."
        return self._camera

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self, init_camera: bool = True, init_grippers: bool = True):
        self._robot = rby1_sdk.create_robot_a(self.address)
        self._robot.connect()
        self.power_on()
        self._connect_configure()

        if init_grippers:
            self._gripper_controller = RainbowGripperController(self._robot)

        if init_camera:
            self._camera = CaptureRS(serial_number=self.RS_SERIAL_NUMBER, auto_close=True)
            self._camera.skip_frames(30)
        else:
            self._camera = None

        self._connected = True
        self.visualizer = None

        def disconnect():
            self.disconnect()
        atexit.register(disconnect)

    def _connect_configure(self):
        for key, value in self.DEFAULT_CONFIGS.items():
            self._robot.set_parameter(key, value)

    def start_visualizer(self):
        self.visualizer = RBY1AServiceVisualizerInterface(self)
        self.visualizer.start()
        import time; time.sleep(1.0)
        if self.has_camera:
            self.get_captures()

    def disconnect(self):
        print('Disconnecting robot...')
        if self._current_stream is not None:
            self._current_stream.cancel()
            self._current_stream.wait()
            self._current_controller = None
            self._current_stream = None

        self.robot.disconnect()
        if self._gripper_controller is not None:
            self._gripper_controller.finish()

    def get_qpos(self) -> Dict[str, np.ndarray]:
        robot_state = self.robot.get_state()
        return_message = {
            '__all__': robot_state.position,
        }
        return_message.update(_get_finegrained_state_vectors(robot_state.position))
        if self._gripper_controller is not None:
            return_message.update({
                'right_gripper': np.array([self.gripper_controller.get_gripper_width('right')]),
                'left_gripper': np.array([self.gripper_controller.get_gripper_width('left')])
            })
        return return_message

    def get_qvel(self) -> Dict[str, np.ndarray]:
        robot_state = self.robot.get_state()
        return_message = {
            '__all__': robot_state.velocity,
        }
        return_message.update(_get_finegrained_state_vectors(robot_state.velocity))
        return return_message

    def get_vector_qpos(self) -> np.ndarray:
        state = np.zeros(28)
        qpos = self.get_qpos()
        qpos_dict = {
            'base': qpos['base'],
            'torso': qpos['torso'],
            'right': qpos['right'],
            'left': qpos['left'],
            'head': qpos['head'],
        }
        if 'right_gripper' in qpos:
            qpos_dict['right_gripper'] = np.array([-qpos['right_gripper'][0] / 2, qpos['right_gripper'][0] / 2])
        if 'left_gripper' in qpos:
            qpos_dict['left_gripper'] = np.array([-qpos['left_gripper'][0] / 2, qpos['left_gripper'][0] / 2])
        return self.pb_robot.set_index_full_joint_state_group_dict(state, qpos_dict)

    def get_captures(self) -> Dict[str, Dict[str, np.ndarray]]:
        color, depth = self.camera.capture()

        self.pb_robot.set_qpos(self.get_vector_qpos())
        head_pose = self.pb_robot.get_link_pose(self.pb_robot.get_ee_link_id('head'))

        T_headlink2_to_cameraoptical = mat2pos_quat_xyzw(self._camera_headlink2_to_cameraoptical)
        cam_pose = frame_mul(head_pose[0], head_pose[1], T_headlink2_to_cameraoptical)
        cam_pose_mat = pos_quat2mat_xyzw(cam_pose[0], cam_pose[1])

        rv = {'head': {
            'color': color,
            'depth': depth,
            'intrinsics': self.camera.intrinsics[0],
            'extrinsics': cam_pose_mat,
            'T_headlink2_to_cameraoptical': self._camera_headlink2_to_cameraoptical,
        }}

        if self.visualizer is not None:
            self.visualizer.update_captures(rv)

        return rv

    def get_captures_compressed(self) -> Dict[str, bytes]:
        color, depth = self.camera.capture()

        self.pb_robot.set_qpos(self.get_vector_qpos())
        head_pose = self.pb_robot.get_link_pose(self.pb_robot.get_ee_link_id('head'))

        T_headlink2_to_cameraoptical = mat2pos_quat_xyzw(self._camera_headlink2_to_cameraoptical)
        cam_pose = frame_mul(head_pose[0], head_pose[1], T_headlink2_to_cameraoptical)
        cam_pose_mat = pos_quat2mat_xyzw(cam_pose[0], cam_pose[1])

        rv = {
            'color': color,
            'depth': depth,
            'intrinsics': self.camera.intrinsics[0],
            'extrinsics': cam_pose_mat,
            'T_headlink2_to_cameraoptical': self._camera_headlink2_to_cameraoptical,
        }

        buf = io.BytesIO()
        np.savez_compressed(buf, **rv)
        return {
            'head': buf.getvalue()
        }

    def maybe_reset_controller(self, controller: str):
        if self._current_controller is None:
            self._current_controller = controller
            self._current_stream = self.robot.create_command_stream(self._priority)
            return self._current_stream
        elif self._current_controller != controller:
            self._current_stream.cancel()
            self._current_stream.wait()
            self._current_controller = None
            self._current_stream = None

            self._current_controller = controller
            self._current_stream = self.robot.create_command_stream(self._priority)
            return self._current_stream

        if self._current_stream.is_done():
            self._current_stream = self.robot.create_command_stream(self._priority)
            return self._current_stream
        return self._current_stream

    def reset_stream(self, controller: str = 'joint_impedance'):
        if self._current_stream is not None:
            self._current_stream.cancel()
            self._current_stream.wait()
            self._current_stream = self.robot.create_command_stream(self._priority)
            self._current_controller = controller

    # def reset_control(self):
    #     if self._current_stream is not None:
    #         self._current_stream.cancel()
    #         self._current_stream.wait()

    #     stream = self.robot.create_command_stream(self._priority + 1)

    def send_command(self, command, timeout):
        try:
            self._current_stream.send_command(command, int(timeout * 1000))
        except RuntimeError as e:
            print(f"Error in sending command: {e}")
            self._current_stream.cancel()
            self._current_stream.wait()
            self._current_stream = self.robot.create_command_stream(self._priority)
            self._current_stream.send_command(command, int(timeout * 1000))

    def move_qpos(self, qpos: Dict[str, np.ndarray], timeout: float = 6, min_time: float = 3, finish_atol: Optional[float] = 1e-3, command_timeout = 1.0):
        # NB(Jiayuan Mao @ 2024/12/16): I don't know why Rainbow guys just set min_time to be 5.0 / 3.0 in their code...
        if len(qpos) == 1 and 'head' in qpos:
            qpos = qpos.copy()
            qpos['torso'] = self.get_qpos()['torso']

        builder = self._build_joint_impedance_command(qpos, min_time=min_time)
        stream = self.maybe_reset_controller('joint_impedance')
        self.send_command(builder, timeout)

        if finish_atol is not None:
            checking_interval = 0.1
            for i in range(int(timeout / checking_interval)):
                time.sleep(checking_interval)
                if stream.is_done():
                    break

                current_qpos = self.get_qpos()
                if _qpos_difference_linf_check(current_qpos, qpos, finishing_atol=finish_atol):
                    break

        print('Move finished', file=sys.stderr)
        builder = self._build_joint_impedance_command(qpos, min_time=60 * 5, control_hold_time=60 * 5)  # 5 minutes
        # builder = self._build_empty_command()  # 5 minutes
        self.send_command(builder, command_timeout)
        print('Send holding command finished', file=sys.stderr)

    def move_qpos_trajectory(self, trajectory: List[Dict[str, np.ndarray]], dt: float = 0.5, min_time_multiplier = 5, first_min_time: Optional[float] = None, command_timeout = 1.0):
        self.maybe_reset_controller('joint_impedance')
        for i, qpos in enumerate(trajectory):
            this_min_time = dt * min_time_multiplier
            this_dt = dt
            if i == 0 and first_min_time is not None:
                assert first_min_time > dt
                this_min_time = first_min_time
                this_dt = first_min_time - dt

            builder = self._build_joint_impedance_command(qpos, this_min_time)
            self._mayby_update_visualizer_qpos_target(qpos)
            self.send_command(builder, command_timeout)
            time.sleep(this_dt)

        print('Move finished', file=sys.stderr)
        qpos = trajectory[-1]
        builder = self._build_joint_impedance_command(qpos, min_time=60 * 5, control_hold_time=60 * 5)  # 5 minutes
        self.send_command(builder, command_timeout)
        print('Send holding command finished', file=sys.stderr)

    def move_gripper_percent_trajectory(self, gripper_name: str, gripper_percent_trajectory: list[float]):
        assert gripper_name in ('left', 'right')
        for i, gripper_open_percent in enumerate(gripper_percent_trajectory):
            self.gripper_controller.set_gripper_percent(gripper_name, gripper_open_percent, wait=True)

    def move_gripper_qpos_trajectory(self, gripper_name: str, trajectory: np.ndarray):
        assert gripper_name in ('left', 'right')
        for i, qpos in enumerate(trajectory):
            self.gripper_controller.set_gripper_open(gripper_name, qpos, wait=True)

    def _build_joint_impedance_command(self, qpos: Dict[str, np.ndarray], min_time: float, control_hold_time: float = 1.0, force_torso_part: bool = False) -> rby1_sdk.RobotCommandBuilder:
        if force_torso_part and 'torso' not in qpos:
            qpos = qpos.copy()
            qpos['torso'] = self.get_qpos()['torso']

        components = rby1_sdk.ComponentBasedCommandBuilder()
        body_command = rby1_sdk.BodyComponentBasedCommandBuilder()
        include_body_command = False

        for key, value in qpos.items():
            if key in ('left', 'right'):
                command = rby1_sdk.JointPositionCommandBuilder()
                command.set_position(value)
                command.set_minimum_time(min_time)
                command.set_command_header(rby1_sdk.CommandHeaderBuilder().set_control_hold_time(control_hold_time))

                if key == 'right':
                    body_command.set_right_arm_command(command)
                else:
                    body_command.set_left_arm_command(command)
                include_body_command = True
            elif key in ('left_cartesian_impedance', 'right_cartesian_impedance'):
                hand = key.split('_')[0]
                command = rby1_sdk.ImpedanceControlCommandBuilder()
                command.set_command_header(rby1_sdk.CommandHeaderBuilder().set_control_hold_time(control_hold_time))
                command.set_reference_link_name("base")
                command.set_link_name(f'ee_{hand}')
                command.set_translation_weight([1000, 1000, 1000]).set_rotation_weight([50, 50, 50])
                command.set_transformation(value)
                if hand == 'right':
                    body_command.set_right_arm_command(command)
                else:
                    body_command.set_left_arm_command(command)
                include_body_command = True
            elif key == 'torso':
                command = rby1_sdk.JointPositionCommandBuilder()
                command.set_position(value)
                command.set_minimum_time(min_time)
                command.set_command_header(rby1_sdk.CommandHeaderBuilder().set_control_hold_time(control_hold_time))
                body_command.set_torso_command(command)
                include_body_command = True
            elif key == 'head':
                command = rby1_sdk.JointPositionCommandBuilder()
                command.set_position(value)
                command.set_minimum_time(min_time)
                command.set_command_header(rby1_sdk.CommandHeaderBuilder().set_control_hold_time(control_hold_time))
                components.set_head_command(command)
            elif key == 'base':
                command = rby1_sdk.JointVelocityCommandBuilder()
                command.set_velocity(value)
                command.set_minimum_time(min_time)
                command.set_command_header(rby1_sdk.CommandHeaderBuilder().set_control_hold_time(control_hold_time))
                components.set_mobility_command(command)
            else:
                raise ValueError(f"Invalid key {key}")

        if include_body_command:
            components.set_body_command(body_command)

        return rby1_sdk.RobotCommandBuilder().set_command(components)

    def _mayby_update_visualizer_qpos_target(self, qpos: Dict[str, np.ndarray]):
        if self.visualizer is not None:
            data = dict()
            time_f = time.time()
            for key, value in qpos.items():
                if key in ('left', 'right', 'torso', 'head'):
                    for i in range(len(value)):
                        data[f'{key.capitalize()}-Joint{i}-Target'] = (time_f, value[i])
                else:
                    raise ValueError(f"Invalid key {key}")
            self.visualizer.queue.put(data)

    def move_zero(self, min_time: float = 10):
        self.move_qpos({
            'base': np.zeros(2),
            'torso': np.zeros(6),
            'right': np.zeros(7),
            'left': np.zeros(7),
            'head': np.zeros(2),
        }, timeout=min_time * 1.5, min_time=min_time)

    def power_on(self):
        _init_robot(self.robot)

    def power_off(self):
        self.robot.power_off('.*')

    def call(self, name, *args, **kwargs):
        return getattr(self, name)(*args, **kwargs)

    def serve_socket(self, name=None, tcp_port=None):
        if name is None:
            name = RBY1AInterfaceClient.DEFAULT_NAME
        if tcp_port is None:
            tcp_port = RBY1AInterfaceClient.DEFAULT_PORTS
        super().serve_socket(name, tcp_port)


class RBY1AServiceVisualizerInterface(object):
    """Create a matplotlib visualizer for joint angles and end-effector poses."""
    def __init__(self, service: RBY1AUPCInterface):
        self.service = service
        self.initialized_event = Event()
        self.stop_event = Event()
        self.publisher = None
        self.queue = queue.Queue()

        def atexit_handler():
            self.stop_event.set()

        atexit.register(atexit_handler)

    def start(self):
        thread = Thread(target=self.mainloop, daemon=True)
        thread.start()
        self.initialized_event.wait()
        thread = Thread(target=self.update_loop, daemon=True)
        thread.start()

    def update_captures(self, captures):
        data = dict()
        for name, capture in captures.items():
            if name != 'head':
                raise ValueError(f"Unknown capture name {name}")
            data['Camera-Head Color'] = (time.time(), capture['color'])
            depth = capture['depth']
            depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
            data['Camera-Head Depth'] = (time.time(), depth_vis[..., ::-1])
        self.queue.put(data)

    def mainloop(self):
        from concepts.hw_interface.robot_state_visualizer.visualizer import RobotStateVisualizer, QueueDescription
        from concepts.hw_interface.robot_state_visualizer.visualizer_multiproc import RobotStateVisualizerPublisher
        self.publisher = RobotStateVisualizerPublisher()

        queues = dict()
        if self.service.has_camera:
            queues[('Camera', 'Camera-Head Color')] = QueueDescription('Camera-Head Color', 'image', 1, group='Camera', width_in_group=6)
            queues[('Camera', 'Camera-Head Depth')] = QueueDescription('Camera-Head Depth', 'image', 1, group='Camera', width_in_group=6)
        for joint_index in range(6):
            queues[('RBY1', f'Torso-Joint{joint_index}')] = QueueDescription(f'Torso-Joint{joint_index}', 'float', 200, group=f'R-Group-Torso', width_in_group=3)
            queues[('RBY1', f'Torso-Joint{joint_index}-Target')] = QueueDescription(f'Torso-Joint{joint_index}-Target', 'float', 200, group=f'R-Group-Torso', attach_to=f'Torso-Joint{joint_index}')
            queues[('RBY1', f'Torso-Joint{joint_index}-Velocity')] = QueueDescription(f'Torso-Joint{joint_index}-Velocity', 'float', 200, group=f'R-Group-Torso', width_in_group=3)
            queues[('RBY1', f'Torso-Joint{joint_index}-Current')] = QueueDescription(f'Torso-Joint{joint_index}-Current', 'float', 200, group=f'R-Group-Torso', attach_to=f'Torso-Joint{joint_index}-Velocity')
            queues[('RBY1', f'Torso-Joint{joint_index}-Torque')] = QueueDescription(f'Torso-Joint{joint_index}-Torque', 'float', 200, group=f'R-Group-Torso', attach_to=f'Torso-Joint{joint_index}-Velocity')
        for joint_index in range(7):
            # visualizer.register_queue('RBY1', f'Right-Joint{joint_index}', 'float', 200, group=f'R-Group-Right', width_in_group=3)
            # visualizer.register_queue('RBY1', f'Left-Joint{joint_index}', 'float', 200, group=f'R-Group-Left', width_in_group=3)
            # visualizer.register_queue('RBY1', f'Right-Joint{joint_index}-Target', 'float', 200, group=f'R-Group-Right', attach_to=f'Right-Joint{joint_index}')
            # visualizer.register_queue('RBY1', f'Left-Joint{joint_index}-Target', 'float', 200, group=f'R-Group-Left', attach_to=f'Left-Joint{joint_index}')
            # visualizer.register_queue('RBY1', f'Right-Joint{joint_index}-Velocity', 'float', 200, group=f'R-Group-Right', width_in_group=3)
            # visualizer.register_queue('RBY1', f'Left-Joint{joint_index}-Velocity', 'float', 200, group=f'R-Group-Left', width_in_group=3)
            # visualizer.register_queue('RBY1', f'Right-Joint{joint_index}-Current', 'float', 200, group=f'R-Group-Right', attach_to=f'Right-Joint{joint_index}')
            # visualizer.register_queue('RBY1', f'Left-Joint{joint_index}-Current', 'float', 200, group=f'R-Group-Left', attach_to=f'Left-Joint{joint_index}')
            # visualizer.register_queue('RBY1', f'Right-Joint{joint_index}-Torque', 'float', 200, group=f'R-Group-Right', attach_to=f'Right-Joint{joint_index}')
            # visualizer.register_queue('RBY1', f'Left-Joint{joint_index}-Torque', 'float', 200, group=f'R-Group-Left', attach_to=f'Left-Joint{joint_index}')
            queues[('RBY1', f'Right-Joint{joint_index}')] = QueueDescription(f'Right-Joint{joint_index}', 'float', 200, group=f'R-Group-Right', width_in_group=3)
            queues[('RBY1', f'Left-Joint{joint_index}')] = QueueDescription(f'Left-Joint{joint_index}', 'float', 200, group=f'R-Group-Left', width_in_group=3)
            queues[('RBY1', f'Right-Joint{joint_index}-Target')] = QueueDescription(f'Right-Joint{joint_index}-Target', 'float', 200, group=f'R-Group-Right', attach_to=f'Right-Joint{joint_index}')
            queues[('RBY1', f'Left-Joint{joint_index}-Target')] = QueueDescription(f'Left-Joint{joint_index}-Target', 'float', 200, group=f'R-Group-Left', attach_to=f'Left-Joint{joint_index}')
            queues[('RBY1', f'Right-Joint{joint_index}-Velocity')] = QueueDescription(f'Right-Joint{joint_index}-Velocity', 'float', 200, group=f'R-Group-Right', width_in_group=3)
            queues[('RBY1', f'Left-Joint{joint_index}-Velocity')] = QueueDescription(f'Left-Joint{joint_index}-Velocity', 'float', 200, group=f'R-Group-Left', width_in_group=3)
            queues[('RBY1', f'Right-Joint{joint_index}-Current')] = QueueDescription(f'Right-Joint{joint_index}-Current', 'float', 200, group=f'R-Group-Right', attach_to=f'Right-Joint{joint_index}-Velocity')
            queues[('RBY1', f'Left-Joint{joint_index}-Current')] = QueueDescription(f'Left-Joint{joint_index}-Current', 'float', 200, group=f'R-Group-Left', attach_to=f'Left-Joint{joint_index}-Velocity')
            queues[('RBY1', f'Right-Joint{joint_index}-Torque')] = QueueDescription(f'Right-Joint{joint_index}-Torque', 'float', 200, group=f'R-Group-Right', attach_to=f'Right-Joint{joint_index}-Velocity')
            queues[('RBY1', f'Left-Joint{joint_index}-Torque')] = QueueDescription(f'Left-Joint{joint_index}-Torque', 'float', 200, group=f'R-Group-Left', attach_to=f'Left-Joint{joint_index}-Velocity')
        for joint_index in range(2):
            # visualizer.register_queue('RBY1', f'Head-Joint{joint_index}', 'float', 200, group=f'R-Group-Head', width_in_group=3)
            # visualizer.register_queue('RBY1', f'Head-Joint{joint_index}-Target', 'float', 200, group=f'R-Group-Head', attach_to=f'Head-Joint{joint_index}')
            queues[('RBY1', f'Head-Joint{joint_index}')] = QueueDescription(f'Head-Joint{joint_index}', 'float', 200, group=f'R-Group-Head', width_in_group=3)
            queues[('RBY1', f'Head-Joint{joint_index}-Target')] = QueueDescription(f'Head-Joint{joint_index}-Target', 'float', 200, group=f'R-Group-Head', attach_to=f'Head-Joint{joint_index}')

        self.publisher.reset(queues)
        self.initialized_event.set()

        while True:
            if self.stop_event.is_set():
                break

            try:
                for _ in range(10):
                    item = self.queue.get(timeout=0.1)
                    self.publisher.publish(item)
            except queue.Empty:
                pass

    def update_loop(self):
        def update():
            time_f = time.time()
            state = self.service.robot.get_state()

            data = dict()
            qpos = _get_finegrained_state_vectors(state.position)
            for group in ['head', 'torso', 'right', 'left']:
                for i in range(len(qpos[group])):
                    data[f'{group.capitalize()}-Joint{i}'] = (time_f, qpos[group][i])
            qvel = _get_finegrained_state_vectors(state.velocity)
            for group in ['torso', 'right', 'left']:
                for i in range(len(qvel[group])):
                    data[f'{group.capitalize()}-Joint{i}-Velocity'] = (time_f, qvel[group][i])
            current = _get_finegrained_state_vectors(state.current)
            for group in ['torso', 'right', 'left']:
                for i in range(len(current[group])):
                    data[f'{group.capitalize()}-Joint{i}-Current'] = (time_f, current[group][i])
            torque = _get_finegrained_state_vectors(state.torque)
            for group in ['torso', 'right', 'left']:
                for i in range(len(torque[group])):
                    data[f'{group.capitalize()}-Joint{i}-Torque'] = (time_f, torque[group][i])
            self.queue.put(data)

            time.sleep(0.1)
            return True

        while True:
            if self.stop_event.is_set():
                return False
            update()


def _init_robot(robot: rby1_sdk.Robot_A):
    if not robot.is_connected():
        print("Robot is not connected")
        exit(1)
    print('Robot is connected')
    power_device = ".*"
    if not robot.is_power_on(power_device):
        rv = robot.power_on(power_device)
        if not rv:
            print("Failed to power on")
            exit(1)
    print('Robot is powered on')
    if not robot.is_servo_on(power_device):
        rv = robot.servo_on(power_device)
        if not rv:
            print("Failed to servo on")
            exit(1)
    print('Robot is servo on')
    robot.reset_fault_control_manager()
    print('Robot reset control manager')
    robot.enable_control_manager()
    print('Robot enabled control manager')


def _get_finegrained_state_vectors(qpos: np.ndarray) -> Dict[str, np.ndarray]:
    batches = {
        'base': qpos[:2],
        'torso': qpos[2:2+6],
        'right': qpos[2+6:2+6+7],
        'left': qpos[2+6+7:2+6+7+7],
        'head': qpos[2+6+7+7:2+6+7+7+2],
    }
    individuals = {
        'right_wheel': qpos[0],
        'left_wheel': qpos[1],
    }
    for i in range(6):
        individuals[f'torso_{i}'] = batches['torso'][i]
    for i in range(7):
        individuals[f'right_{i}'] = batches['right'][i]
        individuals[f'left_{i}'] = batches['left'][i]
    for i in range(2):
        individuals[f'head_{i}'] = batches['head'][i]
    batches.update(individuals)
    return batches


ERROR_THRESHOLD = {
    'head': 0.005,
    'torso': 0.001,
    'right': 0.001,
    'left': 0.001,
    'base': 0.005,
}

def _qpos_difference_linf_check(state, reference, finishing_atol):
    """Compute the L-infinity norm of the difference between the state and the reference."""
    for k, v in reference.items():
        this_tol = max(finishing_atol, ERROR_THRESHOLD.get(k, finishing_atol))
        max_diff = np.max(np.abs(state[k] - v))
        print(f"Checking {k}: {max_diff} vs {this_tol}")
        if max_diff > this_tol:
            return False
    return True

