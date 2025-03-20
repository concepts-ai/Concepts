#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pybullet_manipulator_interface.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/31/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Any, Optional, Union, Tuple, List

import numpy as np

from concepts.dm.crowhat.impl.pybullet.pybullet_planning_world_interface import PyBulletPlanningWorldInterface
from concepts.dm.crowhat.impl.pybullet.pybullet_sim_interfaces import PyBulletPhysicalControllerInterface, PyBulletSimulationControllerInterface
from concepts.dm.crowhat.world.manipulator_interface import (
    RobotArmJointTrajectory, RobotControllerExecutionContext, RobotControllerExecutionFailed, SingleArmControllerInterface,
    SingleGroupMotionPlanningInterface
)
from concepts.dm.crowhat.world.planning_world_interface import AttachmentInfo, PlanningWorldInterface
from concepts.simulator.pybullet.components.panda.panda_robot import PandaRobot
from concepts.simulator.pybullet.components.robot_base import BulletArmRobotBase, BulletMultiChainRobotRobotBase
from concepts.utils.typing_utils import Vec3f, Vec4f, VecNf

__all__ = ['PyBulletSingleArmMotionPlanningInterface', 'PyBulletSubchainMotionPlanningInterface', 'PyBulletSingleArmControllerInterface']


class PyBulletSingleArmMotionPlanningInterface(SingleGroupMotionPlanningInterface):
    def __init__(self, robot: BulletArmRobotBase, planning_world: Optional[PlanningWorldInterface]):
        super().__init__(planning_world)
        self._robot = robot
        self._joints = self._robot.get_joint_ids()
        self._joint_limits = self._robot.get_joint_ids()
        self.set_configuration_space_extra_validation_func(self._robot.is_qpos_valid)

    def get_nr_joints(self) -> int:
        return len(self._joints)

    def get_joint_limits(self):
        return self._robot.get_joint_limits()

    def get_body_id(self) -> int:
        return self._robot.get_body_id()

    def get_ee_link_id(self) -> int:
        return self._robot.get_ee_link_id()

    def get_ee_default_quat(self) -> Vec4f:
        return self._robot.get_ee_default_quat()

    def get_ee_to_tool(self, tool_id: int) -> Tuple[Vec3f, Vec4f]:
        return self._robot.get_ee_to_tool(tool_id)

    def _fk(self, qpos):
        return self._robot.fk(qpos)

    def _ik(self, pos, quat, qpos=None, max_distance=None) -> Optional[np.ndarray]:
        if max_distance is None:
            max_distance = float(1e9)
        return self._robot.ikfast(pos, quat, qpos, max_distance=max_distance, error_on_fail=False)

    def _jacobian(self, qpos):
        return self._robot.get_jacobian(qpos)

    def _mass(self, qpos: np.ndarray) -> np.ndarray:
        return self._robot.get_mass_matrix(qpos)

    def _coriolis(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        return self._robot.get_coriolis_torque(qpos, qvel)

    def _get_qpos(self) -> VecNf:
        return self._robot.get_qpos()

    def _set_qpos(self, qpos: np.ndarray):
        self._robot.set_qpos_with_attached_objects(qpos)

        planning_world: PyBulletPlanningWorldInterface = self.planning_world
        if planning_world.mplib_client is not None:
            planning_world.mplib_client.get_robot(self._robot.get_body_name()).set_qpos(qpos)

            if (index := self._robot.get_attached_object()) is not None:
                name = planning_world.client.world.get_body_name(index)
                planning_world.mplib_client.set_object_pose(name, *planning_world.client.world.get_body_state_by_id(index).get_transformation())

    def _add_attachment(self, body: Union[str, int], link: Optional[int] = None, self_link: Optional[int] = None, ee_to_object: Optional[Tuple[Vec3f, Vec4f]] = None) -> Any:
        assert link is None, 'PyBullet does not support attachment to a specific link (other than the base link of the object).'
        assert self_link is None, 'PyBullet does not support non-EE attachments.'
        if self._robot.get_attached_object() is not None:
            import ipdb; ipdb.set_trace()
        assert self._robot.get_attached_object() is None, 'The robot is already attached to an object.'
        self._robot.attach_object(body, ee_to_object, simulate_gripper=False)
        return None

    def _remove_attachment(self, body: Union[str, int, None], link: Optional[int] = None, self_link: Optional[int] = None):
        self._robot.detach_object()

    def _get_attached_objects(self) -> List[AttachmentInfo]:
        attached_object = self._robot.get_attached_object()
        if attached_object is None:
            return []
        return [AttachmentInfo(
            body_a=self.get_body_id(), link_a=self.ee_link_id, body_b=attached_object, link_b=-1,
            a_to_b=self._robot.get_attached_object_pose_in_ee_frame()
        )]

    def rrt_collision_free(
        self, qpos1: np.ndarray, qpos2: Optional[np.ndarray] = None,
        ignored_collision_bodies: Optional[List[Union[str, int]]] = None,
        smooth_fine_path: bool = False, disable_rendering: bool = True, **kwargs
    ) -> Tuple[bool, Optional[List[np.ndarray]]]:
        if disable_rendering:
            assert isinstance(self.planning_world, PyBulletPlanningWorldInterface), 'The planning world must be a PyBulletPlanningWorldInterface.'
            planning_world: PyBulletPlanningWorldInterface = self.planning_world
            client = planning_world.client
            with client.disable_rendering(suppress_stdout=False):
                return super().rrt_collision_free(qpos1, qpos2, ignored_collision_bodies, smooth_fine_path, **kwargs)
        return super().rrt_collision_free(qpos1, qpos2, ignored_collision_bodies, smooth_fine_path, **kwargs)


class PyBulletSubchainMotionPlanningInterface(SingleGroupMotionPlanningInterface):
    """A motion planning interface for a subchain of a robot that mimics the behavior of a "single-armed" robot."""

    def __init__(self, robot: BulletMultiChainRobotRobotBase, ee_link_name: str, planning_world: Optional[PlanningWorldInterface]):
        super().__init__(planning_world)
        self._robot = robot
        self._joints = self._robot.get_joint_ids()
        self._joint_limits = self._robot.get_joint_limits()
        self._ee_link_name = ee_link_name
        self._ee_link_id = self._robot.world.get_link_index_with_body(self._robot.get_body_id(), ee_link_name)
        self.set_configuration_space_extra_validation_func(self._robot.is_qpos_valid)

    def get_nr_joints(self) -> int:
        return len(self._joints)

    def get_joint_limits(self):
        return self._joint_limits

    def get_body_id(self) -> int:
        return self._robot.get_body_id()

    def get_ee_link_id(self) -> int:
        return self._ee_link_id

    def get_ee_default_quat(self) -> Vec4f:
        return self._robot.get_ee_link_default_quat(self._ee_link_id)

    def _fk(self, qpos):
        return self._robot.fk(qpos, link_name_or_id=self._ee_link_id)

    def _ik(self, pos, quat, qpos=None, max_distance=None) -> Optional[np.ndarray]:
        if max_distance is None:
            max_distance = float(1e9)
        return self._robot.ik_tracik(pos, quat, self._ee_link_id, last_qpos=qpos, max_distance=max_distance)
        # return self._robot.ik_scipy(pos, quat, group_name=self._group_name, last_qpos=qpos, max_distance=max_distance)

    def _jacobian(self, qpos):
        return self._robot.get_jacobian(qpos, link_id=self._ee_link_id)

    def _mass(self, qpos: np.ndarray) -> np.ndarray:
        return self._robot.get_mass_matrix(qpos)

    def _coriolis(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        return self._robot.get_coriolis_torque(qpos, qvel)

    def _get_qpos(self) -> VecNf:
        return self._robot.get_qpos()

    def _set_qpos(self, qpos: np.ndarray):
        return self._robot.set_qpos_with_attached_objects(qpos)

    def _add_attachment(self, body: Union[str, int], link: Optional[int] = None, self_link: Optional[int] = None, ee_to_object: Optional[Tuple[Vec3f, Vec4f]] = None) -> Any:
        raise NotImplementedError('Subchain motion planning interface does not support attachments.')

    def _remove_attachment(self, body: Union[str, int], link: Optional[int] = None, self_link: Optional[int] = None):
        raise NotImplementedError('Subchain motion planning interface does not support attachments.')

    def _get_attached_objects(self) -> List[AttachmentInfo]:
        attached_object = list()
        for ee_id in self._robot.gripper_constraints:
            if self._robot.get_attached_object(ee_id) is not None:
                attached_object.append(AttachmentInfo(
                    body_a=self.get_body_id(), link_a=ee_id, body_b=self._robot.get_attached_object(ee_id), link_b=-1,
                    a_to_b=self._robot.get_attached_object_pose_in_ee_frame(ee_id)
                ))
        return attached_object


class PyBulletSingleArmControllerInterface(SingleArmControllerInterface):
    """This implementation has been deprecated."""

    def __init__(self, panda_robot, timeout_multiplier: float = 1.0):
        super().__init__()
        self._panda_robot = panda_robot
        self.timeout_multiplier = timeout_multiplier
        self.set_default_parameters('move_grasp', move_home_timeout=30, trajectory_timeout=50)
        self.set_default_parameters('move_place', trajectory_timeout=50)

    @property
    def panda_robot(self) -> PandaRobot:
        return self._panda_robot

    def get_qpos(self) -> np.ndarray:
        return self.panda_robot.get_qpos()

    def get_qvel(self) -> np.ndarray:
        raise NotImplementedError('qvel for panda robot is not implemented.')

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.panda_robot.get_ee_pose()

    def move_home(self, **kwargs) -> None:
        self.get_update_default_parameters('move_home', kwargs)
        with RobotControllerExecutionContext('move_home', **kwargs) as ctx:
            ctx.monitor(self.panda_robot.move_home(**kwargs))

    def move_qpos(self, qpos: np.ndarray, **kwargs) -> None:
        self.get_update_default_parameters('move_qpos', kwargs)
        with RobotControllerExecutionContext('move_qpos', qpos, **kwargs) as ctx:
            ctx.monitor(self.panda_robot.move_qpos(qpos, **kwargs))

    def move_pose(self, pos: np.ndarray, quat: np.ndarray, **kwargs) -> None:
        self.get_update_default_parameters('move_pose', kwargs)
        with RobotControllerExecutionContext('move_pose', pos, quat, **kwargs) as ctx:
            ctx.monitor(self.panda_robot.move_pose(pos, quat, **kwargs))

    def move_qpos_trajectory(self, trajectory: RobotArmJointTrajectory, **kwargs) -> None:
        self.get_update_default_parameters('move_qpos_trajectory', kwargs)
        with RobotControllerExecutionContext('move_qpos_trajectory', trajectory, **kwargs) as ctx:
            ctx.monitor(self.panda_robot.move_qpos_path_v2(trajectory, **kwargs))

    def open_gripper(self, **kwargs) -> None:
        self.get_update_default_parameters('open_gripper', kwargs)
        with RobotControllerExecutionContext('open_gripper', **kwargs) as ctx:
            ctx.monitor(self.panda_robot.open_gripper_free(**kwargs))

    def close_gripper(self, **kwargs) -> None:
        self.get_update_default_parameters('close_gripper', kwargs)
        with RobotControllerExecutionContext('close_gripper', **kwargs) as ctx:
            ctx.monitor(self.panda_robot.close_gripper_free(**kwargs))

    def grasp(self, width: float = 0.05, force: float = 40, **kwargs) -> None:
        # NB(Jiayuan Mao @ 2024/03/30): width and force parameters are ignored in the PyBullet-based implementation because it uses magic attachments.
        self.get_update_default_parameters('grasp', kwargs)
        with RobotControllerExecutionContext('grasp', **kwargs) as ctx:
            ctx.monitor(self.panda_robot.grasp(**kwargs))

    def move_grasp(self, approaching_trajectory: RobotArmJointTrajectory, width: float = 0.05, force: float = 40, **kwargs):
        self.get_update_default_parameters('move_grasp', kwargs)
        move_home_timeout = kwargs['move_home_timeout']
        trajectory_timeout = kwargs['trajectory_timeout']

        try:
            self.move_home(timeout=move_home_timeout)
            self._smooth_move_qpos_trajectory(approaching_trajectory, timeout=trajectory_timeout)
            self.grasp(width, force)
            self._smooth_move_qpos_trajectory(approaching_trajectory[:-5][::-1], timeout=trajectory_timeout)
        except RobotControllerExecutionFailed:
            raise

    def move_place(self, placing_trajectory: RobotArmJointTrajectory, **kwargs):
        self.get_update_default_parameters('move_place', kwargs)
        trajectory_timeout = kwargs['trajectory_timeout']

        try:
            self._smooth_move_qpos_trajectory(placing_trajectory, timeout=trajectory_timeout)
            self.open_gripper()
            self.move_home()
        except RobotControllerExecutionFailed:
            raise

    def _smooth_move_qpos_trajectory(self, qt: RobotArmJointTrajectory, timeout: float = 10):
        robot = self.panda_robot
        qt = qt.copy()
        qt.insert(0, robot.get_qpos())

        smooth_qt = [qt[0]]
        cspace = robot.get_configuration_space()
        for qpos1, qpos2 in zip(qt[:-1], qt[1:]):
            smooth_qt.extend(cspace.gen_path(qpos1, qpos2)[1][1:])

        with RobotControllerExecutionContext('move_qpos_trajectory', smooth_qt) as ctx:
            ctx.monitor(robot.move_qpos_path_v2(smooth_qt, timeout=timeout * self.timeout_multiplier))

    def attach_physical_interface(self, physical_interface: PyBulletPhysicalControllerInterface):
        physical_interface.register_controller('move_home', self.move_home)
        physical_interface.register_controller('move_qpos', self.move_qpos)
        physical_interface.register_controller('move_pose', self.move_pose)
        physical_interface.register_controller('move_qpos_trajectory', self.move_qpos_trajectory)
        physical_interface.register_controller('open_gripper', self.open_gripper)
        physical_interface.register_controller('close_gripper', self.close_gripper)
        physical_interface.register_controller('grasp', self.grasp)

        physical_interface.register_controller('ctl_grasp', self.move_grasp)
        physical_interface.register_controller('ctl_place', self.move_place)

    def attach_simulation_interface(self, simulation_interface: PyBulletSimulationControllerInterface):
        simulation_interface.register_controller('move_home', self.move_home)
        simulation_interface.register_controller('move_qpos', self.move_qpos)
        simulation_interface.register_controller('move_pose', self.move_pose)
        simulation_interface.register_controller('move_qpos_trajectory', self.move_qpos_trajectory)
        simulation_interface.register_controller('open_gripper', self.open_gripper)
        simulation_interface.register_controller('close_gripper', self.close_gripper)
        simulation_interface.register_controller('grasp', self.grasp)

        simulation_interface.register_controller('ctl_grasp', self.move_grasp)
        simulation_interface.register_controller('ctl_place', self.move_place)
