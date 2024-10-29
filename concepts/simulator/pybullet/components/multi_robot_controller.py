#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : multi_robot_controller.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/08/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional, Sequence, Tuple, List, NamedTuple, Callable

import numpy as np

from jacinle.utils.enum import JacEnum
from concepts.simulator.pybullet.components.component_base import BulletComponent
from concepts.simulator.pybullet.components.robot_base import BulletArmRobotBase
from concepts.math.interpolation_utils import gen_linear_spline, get_next_target_linear_spline


class _ControlCommand(NamedTuple):
    func: Callable
    kwargs: dict


class RecordedControlCommandType(JacEnum):
    BEGIN_SYNC_CONTEXT = 'begin_sync_context'
    END_SYNC_CONTEXT = 'end_sync_context'
    DO = 'do'
    DO_SYNCHRONIZED_QPOS_TRAJECTORIES_AUTO_SMOOTHING = 'do_synchronized_qpos_trajectories'


class RecordedControlDoCommand(NamedTuple):
    robot_index: int
    action_name: str
    kwargs: dict


class RecordedControlDoSynchronizedQposTrajectoriesCommand(NamedTuple):
    trajectories: Sequence[Sequence[np.ndarray]]
    step_size: float
    gains: float
    atol: float
    timeout: float


class RecordedControlCommand(NamedTuple):
    type: RecordedControlCommandType
    payload: Optional[NamedTuple] = None


class MultiRobotController(BulletComponent):
    def __init__(self, robots: Sequence[BulletArmRobotBase]):
        assert len(robots) > 0

        super().__init__(robots[0].client)
        self.robots = robots
        self.current_ctx = None
        self.recording_enabled = False
        self.recorded_commands = list()

    ACTION_NAME_MAPPING = {
        'move_qpos': 'move_qpos_set_control',
        'move_qpos_path_v2': 'move_qpos_path_v2_set_control',
        'move_cartesian_trajectory': 'move_cartesian_trajectory_set_control',
        'open_gripper_free': 'open_gripper_free_set_control',
        'close_gripper_free': 'close_gripper_free_set_control',
        'grasp': 'grasp_set_control',
    }

    # Instance members
    robots: Sequence[BulletArmRobotBase]
    """The sequence of robot arms controlled by this MultiRobotController."""

    current_ctx: Optional['MultiRobotControllerContext']
    """The current synchronization context, if any."""

    recording_enabled: bool
    """Whether command recording is enabled."""

    recorded_commands: List[RecordedControlCommand]
    """The list of recorded commands when recording is enabled."""

    def enable_recording(self):
        self.recording_enabled = True

    def disable_recoding(self):
        self.recording_enabled = False

    def get_concat_qpos(self):
        return np.concatenate([robot.get_qpos() for robot in self.robots])

    def make_sync_context(self):
        return MultiRobotControllerContext(self)

    def do(self, robot_index: int, action_name: str, **kwargs) -> _ControlCommand:
        """Execute an action on a specific robot.

        Args:
            robot_index: index of the robot to perform the action on.
            action_name: name of the action to perform.
            **kwargs: additional keyword arguments for the action.

        Returns:
            a control command object representing the action.

        Raises:
            AssertionError: If the robot index is out of range.
        """
        assert 0 <= robot_index < len(self.robots)

        if self.recording_enabled:
            self.recorded_commands.append(RecordedControlCommand(
                RecordedControlCommandType.DO,
                RecordedControlDoCommand(robot_index, action_name, kwargs)
            ))

        cmd = _ControlCommand(
            getattr(self.robots[robot_index], self.ACTION_NAME_MAPPING[action_name]),
            kwargs
        )
        if self.current_ctx is not None:
            if robot_index not in self.current_ctx.commands:
                self.current_ctx.commands[robot_index] = list()
            self.current_ctx.commands[robot_index].append(cmd)
        return cmd

    def do_synchronized_qpos_trajectories_auto_smoothing(
        self, trajectories: Sequence[Sequence[np.ndarray]],
        step_size: float = 1, gains: float = 0.3, atol: float = 0.03, timeout: float = 20, verbose: bool = False
    ):
        """Execute a synchronized qpos trajectory auto-smoothing action on multiple robots.

        Args:
            trajectories: a sequence of qpos trajectories for each robot.
            step_size: the step size for the auto-smoothing.
            gains: the gains for the auto-smoothing.
            atol: the absolute tolerance for the auto-smoothing.
            timeout: the timeout for the auto-smoothing.
            verbose: whether to print verbose output.

        Raises:
            AssertionError: If the number of trajectories does not match the number of robots.
            AssertionError: If the length of the trajectories does not match.
        """
        assert len(trajectories) == len(self.robots) > 0
        nr_length = len(trajectories[0])
        for i in range(len(trajectories)):
            assert len(trajectories[i]) == nr_length

        if self.recording_enabled:
            self.recorded_commands.append(RecordedControlCommand(
                RecordedControlCommandType.DO_SYNCHRONIZED_QPOS_TRAJECTORIES_AUTO_SMOOTHING,
                RecordedControlDoSynchronizedQposTrajectoriesCommand(trajectories, step_size, gains, atol, timeout)
            ))

        MultiRobotMoveTrajectoryAutoSmoothing(self, self.robots, trajectories).move(
            step_size=step_size, gains=gains, atol=atol, timeout=timeout, verbose=verbose
        )

    def do_synchronized_ee_pose_trajectories(self, trajectories: Sequence[Sequence[Tuple[np.ndarray, np.ndarray]]]):
        """Execute a synchronized ee pose trajectory action on multiple robots.

        Args:
            trajectories: a sequence of ee pose trajectories for each robot.

        Raises:
            AssertionError: If the number of trajectories does not match the number of robots.
            AssertionError: If the length of the trajectories does not match.
        """
        assert len(trajectories) == len(self.robots) > 0
        nr_length = len(trajectories[0])
        for i in range(len(trajectories)):
            assert len(trajectories[i]) == nr_length

        if self.recording_enabled:
            raise NotImplementedError('Recording synchronized ee pose trajectories is not supported yet.')

        MultiRobotMoveEEPoseTrajectory(self, self.robots, trajectories).move()

    def stable_reset(self, nr_steps=10):
        """Reset the robots to a stable state by holding their current positions.

        Args:
            nr_steps: the number of steps to hold the positions.
        """
        robot_qposes = [robot.get_qpos() for robot in self.robots]
        for i in range(nr_steps):
            for j, robot in enumerate(self.robots):
                robot.set_full_hold_position_control(robot_qposes[j])
            self.client.step()

    def replay(self, commands):
        """Replay a sequence of recorded commands.

        Args:
            commands: a sequence of recorded commands to replay.

        Raises:
            AssertionError: If the recording is enabled.
        """
        assert self.recording_enabled is False, 'Replay is not allowed when recording is enabled.'
        for cmd in commands:
            if cmd.type is RecordedControlCommandType.DO:
                robot_index = cmd.payload.robot_index
                action_name = cmd.payload.action_name
                kwargs = cmd.payload.kwargs
                self.do(robot_index, action_name, **kwargs)
            elif cmd.type is RecordedControlCommandType.DO_SYNCHRONIZED_QPOS_TRAJECTORIES_AUTO_SMOOTHING:
                trajectories = cmd.payload.trajectories
                step_size = cmd.payload.step_size
                gains = cmd.payload.gains
                atol = cmd.payload.atol
                timeout = cmd.payload.timeout
                self.do_synchronized_qpos_trajectories_auto_smoothing(trajectories, step_size=step_size, gains=gains, atol=atol, timeout=timeout)
            elif cmd.type is RecordedControlCommandType.BEGIN_SYNC_CONTEXT:
                self.make_sync_context().begin()
            elif cmd.type is RecordedControlCommandType.END_SYNC_CONTEXT:
                self.current_ctx.end()
            else:
                raise ValueError(f'Unknown command type: {cmd.type}')


class MultiRobotControllerContext(object):
    """A context for controlling multiple robots synchronously."""

    def __init__(self, controller: MultiRobotController):
        """Initialize the synchronization context.

        Args:
            controller: the controller that manages the robots.
        """
        self.controller = controller
        self.commands = dict()

    def begin(self):
        self.controller.current_ctx = self
        self.commands = dict()

        if self.controller.recording_enabled:
            self.controller.recorded_commands.append(RecordedControlCommand(RecordedControlCommandType.BEGIN_SYNC_CONTEXT))

        return self

    def __enter__(self):
        return self.begin()

    def end(self):
        if self.controller.recording_enabled:
            self.controller.recorded_commands.append(RecordedControlCommand(RecordedControlCommandType.END_SYNC_CONTEXT))

        self.run_commands()
        self.controller.current_ctx = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()

    def run_commands(self):
        current_iterators = dict()
        current_iterator_indices = dict()
        for robot_index, cmds in self.commands.items():
            current_iterator_indices[robot_index] = 0
            current_iterators[robot_index] = cmds[0].func(**cmds[0].kwargs)

        dones = [False] * len(self.controller.robots)
        qposes = [None] * len(self.controller.robots)

        for i in range(len(dones)):
            if i not in self.commands:
                dones[i] = True
                qposes[i] = self.controller.robots[i].get_qpos()

        if all(dones):
            return

        timestep = 0
        while True:
            timestep += 1
            for i, done in enumerate(dones):
                # print(f'{timestep=}:: {i=} {done=}')
                if done:
                    self.controller.robots[i].set_full_hold_position_control(qposes[i])
                else:
                    try:
                        next(current_iterators[i])
                    except StopIteration:
                        if current_iterator_indices[i] + 1 < len(self.commands[i]):
                            current_iterator_indices[i] += 1
                            cmd = self.commands[i][current_iterator_indices[i]]
                            current_iterators[i] = cmd.func(**cmd.kwargs)
                        else:
                            dones[i] = True
                            qposes[i] = self.controller.robots[i].get_qpos()
            self.controller.client.step()
            if all(dones):
                break


class MultiRobotMoveTrajectoryAutoSmoothing(object):
    def __init__(self, controller: MultiRobotController, robots: Sequence[BulletArmRobotBase], qpos_trajectories: Sequence[Sequence[np.ndarray]]):
        self.controller = controller
        self.robots = robots
        self.qpos_trajectories = qpos_trajectories

        self.concat_qpos_trajectories = self._dedup_qpos_trajectory(np.concatenate([t for t in qpos_trajectories], axis=1))  # (nr_steps, nr_joints * nr_robots)
        self.q_start_indices = np.cumsum([0] + [len(t[0]) for t in qpos_trajectories])[:-1]
        self.q_lengths = [len(t[0]) for t in qpos_trajectories]

    @property
    def client(self):
        return self.robots[0].client

    def _dedup_qpos_trajectory(self, qpos_trajectory):
        qpos_trajectory = np.array(qpos_trajectory)
        qpos_trajectory_dedup = list()
        last_qpos = None
        for qpos in qpos_trajectory:
            if qpos is None:
                continue
            if last_qpos is None or np.linalg.norm(qpos - last_qpos, ord=2) > 0.01:
                qpos_trajectory_dedup.append(qpos)
                last_qpos = qpos
        qpos_trajectory = np.array(qpos_trajectory_dedup)
        return qpos_trajectory

    def set_control(
        self, step_size: float = 1, gains: float = 0.3,
        atol: float = 0.03, timeout: float = 20,
        verbose: bool = False,
    ):
        # spl = gen_cubic_spline(qpos_trajectory)
        spl = gen_linear_spline(self.concat_qpos_trajectories)
        prev_qpos = None
        prev_qpos_not_moving = 0
        next_id = None

        for _ in self.client.timeout(timeout):
            current_qpos = self.controller.get_concat_qpos()
            # next_target = get_next_target_cubic_spline(spl, current_qpos, step_size, qpos_trajectory)
            next_id, next_target = get_next_target_linear_spline(
                spl, current_qpos, step_size,
                minimum_x=next_id - step_size + 0.2 if next_id is not None else None
            )
            last_norm = np.linalg.norm(self.concat_qpos_trajectories[-1] - current_qpos, ord=1)

            if verbose:
                print('last_norm', last_norm)
            if prev_qpos is not None:
                last_moving_dist = np.linalg.norm(prev_qpos - current_qpos, ord=1)
                if last_moving_dist < 0.001:
                    prev_qpos_not_moving += 1
                else:
                    prev_qpos_not_moving = 0
                if prev_qpos_not_moving > 10:
                    if last_norm < atol * 10:
                        return True
                    else:
                        print('Not moving for a long time (10 steps).')
                        return False
            prev_qpos = current_qpos

            if last_norm < atol:
                print('Last norm is smaller than atol.', last_norm, atol)
                return True

            for i, robot in enumerate(self.robots):
                robot.set_arm_joint_position_control(
                    next_target[self.q_start_indices[i]:self.q_start_indices[i] + self.q_lengths[i]],
                    gains=gains, set_gripper_control=True
                )
            yield

        return False

    def move(self, step_size: float = 1, gains: float = 0.3, atol: float = 0.03, timeout: float = 20, verbose: bool = False):
        try:
            for _ in self.set_control(step_size=step_size, gains=gains, atol=atol, timeout=timeout, verbose=verbose):
                self.client.step()
            return True
        except StopIteration as e:
            return e.value


class MultiRobotMoveEEPoseTrajectory(object):
    def __init__(self, controller: MultiRobotController, robots: Sequence[BulletArmRobotBase], ee_pose_trajectories: Sequence[Sequence[Tuple[np.ndarray, np.ndarray]]], verbose: bool = False):
        self.controller = controller
        self.robots = robots
        self.ee_pose_trajectories = ee_pose_trajectories
        self.verbose = verbose

    @property
    def client(self):
        return self.robots[0].client

    def set_control(self):
        for index in range(len(self.ee_pose_trajectories[0])):
            for i, robot in enumerate(self.robots):
                robot.set_ee_impedance_control(*self.ee_pose_trajectories[i][index], verbose=self.verbose)
            yield

    def move(self):
        try:
            for _ in self.set_control():
                self.client.step()
            return True
        except StopIteration as e:
            return e.value

