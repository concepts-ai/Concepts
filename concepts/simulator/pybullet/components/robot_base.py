#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : robot_base.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/24/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""This file contains basic classes for defining robots and their action primitives."""

import contextlib

import numpy as np
from typing import Optional, Union, Iterable, Sequence, Tuple, List, Dict, Callable

import pybullet

from jacinle.logging import get_logger
from jacinle.utils.enum import JacEnum
from jacinle.utils.meta import cond_with

from concepts.algorithm.configuration_space import BoxConfigurationSpace, CollisionFreeProblemSpace
from concepts.algorithm.rrt.rrt import birrt
from concepts.math.range import Range
from concepts.math.rotationlib_xyzw import quat_mul, quat_conjugate
from concepts.math.frame_utils_xyzw import get_transform_a_to_b, calc_ee_quat_from_directions
from concepts.utils.typing_utils import Vec3f, Vec4f

from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.pybullet.world import BodyFullStateSaver
from concepts.simulator.pybullet.components.component_base import BulletComponent

logger = get_logger(__file__)

__all__ = ['BulletGripperBase', 'GripperObjectIndices', 'IKMethod', 'BulletRobotBase', 'BulletArmRobotBase', 'BulletRobotActionPrimitive']


class BulletGripperBase(BulletComponent):
    """Base gripper class."""

    def __init__(self, client: BulletClient):
        super().__init__(client)
        self.activated = False

    def step(self):
        """This function can be used to create gripper-specific behaviors."""
        return

    def activate(self, objects):
        return

    def release(self):
        return


GripperObjectIndices = Dict[str, List[int]]


class IKMethod(JacEnum):
    PYBULLET = 'pybullet'
    IKFAST = 'ikfast'


class BulletRobotBase(BulletComponent):
    def __init__(self, client: BulletClient, body_name: Optional[str] = None, gripper_objects: Optional[GripperObjectIndices] = None, current_interface='pybullet', ik_method: Union[str, IKMethod] = 'pybullet'):
        super().__init__(client)
        self.body_name = body_name if body_name is not None else self.__class__.__name__
        self.gripper_objects = gripper_objects
        if self.gripper_objects is None:
            self.gripper_objects = self.client.w.body_groups

        self.current_interface = current_interface
        self.primitive_actions: Dict[Tuple[str, str], Callable] = dict()
        self.ik_method = IKMethod.from_string(ik_method)
        self.warnings_suppressed = False

        self._cspace = None
        self._cfree_default_pspace = None

    @contextlib.contextmanager
    def suppress_warnings(self):
        backup = self.warnings_suppressed
        self.warnings_suppressed = True
        yield
        self.warnings_suppressed = backup

    def set_suppress_warnings(self, value: bool = True):
        self.warnings_suppressed = value

    def register_action(self, name: str, func: Callable, interface: Optional[str] = None) -> None:
        """Register an action primitive.

        Args:
            name: name of the action primitive.
            func: function that implements the action primitive.
            interface: interface of the action primitive. Defaults to None.
                If None, the action primitive is registered for the current interface.

        Raises:
            ValueError: If the action primitive is already registered.
        """
        if interface is None:
            interface = self.current_interface
        if (name, interface) in self.primitive_actions:
            raise ValueError(f'Action primitive {name} for interface {interface} already registered.')
        self.primitive_actions[(interface, name)] = func

    def do(self, action_name: str, *args, **kwargs) -> bool:
        """Execute an action primitive.

        Args:
            action_name: Name of the action primitive.

        Returns:
            bool: True if the action primitive is successful.
        """
        assert (self.current_interface, action_name) in self.primitive_actions, f'Action primitive {action_name} for interface {self.current_interface} not registered.'
        return self.primitive_actions[(self.current_interface, action_name)](*args, **kwargs)

    def set_ik_method(self, ik_method: Union[str, IKMethod]):
        self.ik_method = IKMethod.from_string(ik_method)

    def get_body_id(self) -> int:
        """Get the pybullet body ID of the robot.

        Returns:
            int: Body ID of the robot.
        """

        raise NotImplementedError()

    def get_joint_ids(self) -> Sequence[int]:
        """Get the pybullet joint IDs of the robot.

        Returns:
            Sequence[int]: Joint IDs of the robot.
        """
        raise NotImplementedError()

    def get_home_qpos(self) -> np.ndarray:
        """Get the home joint configuration."""
        raise NotImplementedError()

    def reset_home_qpos(self):
        """Reset the home joint configuration."""
        raise NotImplementedError()

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the joint limits.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Lower and upper joint limits.
        """
        body_id = self.get_body_id()
        joint_info = [self.client.w.get_joint_info_by_id(body_id, i) for i in self.get_joint_ids()]
        lower_limits = np.array([joint.joint_lower_limit for joint in joint_info])
        upper_limits = np.array([joint.joint_upper_limit for joint in joint_info])
        return lower_limits, upper_limits

    def get_dof(self) -> int:
        """Get the total degrees of freedom of the robot.

        Returns:
            int: Total degrees of freedom.
        """
        return len(self.get_joint_ids())

    def get_qpos(self) -> np.ndarray:
        """Get the current joint configuration.

        Returns:
            np.ndarray: Current joint configuration.
        """
        return self.client.w.get_batched_qpos_by_id(self.get_body_id(), self.get_joint_ids())

    def set_qpos(self, qpos: np.ndarray) -> None:
        """Set the joint configuration.

        Args:
            qpos: Joint configuration.
        """
        self.client.w.set_batched_qpos_by_id(self.get_body_id(), self.get_joint_ids(), qpos)

    def set_qpos_with_holding(self, qpos: np.ndarray) -> None:
        """Set the joint configuration with the gripper holding the object.

        Args:
            qpos: Joint configuration.
        """
        return self.set_qpos(qpos)

    def get_qvel(self) -> np.ndarray:
        """Get the current joint velocity.

        Returns:
            np.ndarray: Current joint velocity.
        """
        return self.client.w.get_batched_qvel_by_id(self.get_body_id(), self.get_joint_ids())

    def get_body_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the pose of the robot body.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 3D position and 4D quaternion of the robot body.
        """
        state = self.client.w.get_body_state_by_id(self.get_body_id())
        return np.array(state.position), np.array(state.orientation)


class BulletArmRobotBase(BulletRobotBase):
    """Base class for arm robots, such as UR5, Panda, etc.

    In this class, we assume that the robot joints are composed of two parts: the arm joints and the gripper joints.
    All joint IDs and joint configurations functions are based on arm joints. To handle the "full" joint configuration,
    we provide an additional set of functions to get and set the full joint configuration, including the gripper joints.
    """

    def __init__(
        self,
        client: BulletClient,
        body_name: Optional[str] = None,
        gripper_objects: Optional[GripperObjectIndices] = None,
        current_interface='pybullet',
        ik_method: Union[str, IKMethod] = 'pybullet',
        use_magic_gripper: bool = True
    ):
        super().__init__(client, body_name, gripper_objects, current_interface, ik_method)
        self.use_magic_gripper = use_magic_gripper
        self.gripper_constraint = None

    def get_full_joint_ids(self) -> Sequence[int]:
        """Get the pybullet joint IDs of the robot, including the gripper joints.

        Returns:
            Sequence[int]: Joint IDs of the robot.
        """
        raise NotImplementedError()

    def get_full_dof(self) -> int:
        """Get the total degrees of freedom of the robot, including the gripper joints.

        Returns:
            int: Total degrees of freedom.
        """
        return len(self.get_full_joint_ids())

    def get_full_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the joint limits of the robot arm, including the gripper joints.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Lower and upper joint limits.
        """
        body = self.get_body_id()
        joint_info = [self.client.w.get_joint_info_by_id(body, i) for i in self.get_full_joint_ids()]
        lower_limits = np.array([joint.joint_lower_limit for joint in joint_info])
        upper_limits = np.array([joint.joint_upper_limit for joint in joint_info])
        return lower_limits, upper_limits

    def get_full_qpos(self) -> np.ndarray:
        """Get the current joint configuration.

        Returns:
            np.ndarray: Current joint configuration.
        """
        return self.client.w.get_batched_qpos_by_id(self.get_body_id(), self.get_full_joint_ids())

    def get_full_home_qpos(self) -> np.ndarray:
        """Get the home joint configuration."""
        raise NotImplementedError()

    def set_full_qpos(self, qpos: np.ndarray) -> None:
        """Set the joint configuration.

        Args:
            qpos: Joint configuration.
        """
        self.client.w.set_batched_qpos_by_id(self.get_body_id(), self.get_full_joint_ids(), qpos)

    def get_full_qvel(self) -> np.ndarray:
        """Get the current joint velocity.

        Returns:
            np.ndarray: Current joint velocity.
        """
        return self.client.w.get_batched_qvel_by_id(self.get_body_id(), self.get_full_joint_ids())

    def get_ee_link_id(self) -> int:
        """Get the pybullet link ID of the robot end effector.

        Returns:
            int: Link ID of the robot end effector.
        """
        raise NotImplementedError()

    def get_ee_home_pos(self) -> np.ndarray:
        """Get the home position of the end effector."""
        raise NotImplementedError()

    def get_ee_home_quat(self) -> np.ndarray:
        """Get the home orientation of the end effector."""
        raise NotImplementedError()

    def get_ee_pose(self, fk: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Get the pose of the end effector.

        Args:
            fk: whether to run forward kinematics to re-compute the pose.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 3D position and 4D quaternion of the end effector.
        """
        state = self.client.w.get_link_state_by_id(self.get_body_id(), self.get_ee_link_id(), fk=fk)
        return np.array(state.position), np.array(state.orientation)

    def get_ee_velocity(self, fk: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Get the velocity of the end effector.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 3D linear velocity and 3D angular velocity of the end effector.
        """
        state = self.client.w.get_link_state_by_id(self.get_body_id(), self.get_ee_link_id(), fk=fk)
        return np.array(state.linear_velocity), np.array(state.angular_velocity)

    def set_ee_pose(self, pos: np.ndarray, quat: np.ndarray) -> bool:
        """Set the pose of the end effector by inverse kinematics. Return True if the IK solver succeeds.

        Args:
            pos: 3D position of the end effector.
            quat: 4D quaternion of the end effector.

        Returns:
            bool: True if the IK solver succeeds.
        """
        qpos = self.ik(pos, quat)
        if qpos is None:
            return False
        self.set_qpos(qpos)
        return True

    def get_ee_to_tool(self, tool_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get the pose of the tool frame relative to the end effector frame.

        Args:
            tool_id: ID of the tool frame.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 3D position and 4D quaternion of the tool frame.
        """

        robot_pos, robot_quat = self.world.get_link_state_by_id(self.get_body_id(), self.get_ee_link_id(), fk=True).get_transformation()
        tool_pos, tool_quat = self.world.get_body_state_by_id(tool_id).get_transformation()
        return get_transform_a_to_b(robot_pos, robot_quat, tool_pos, tool_quat)

    def get_ee_default_quat(self) -> np.ndarray:
        """Get the default orientation of the end effector."""
        raise NotImplementedError()

    def get_ee_quat_from_vectors(self, u: Vec3f = (-1., 0., 0.), v: Vec3f = (1., 0., 0.)) -> Vec4f:
        """Compute the quaternion from two directions (the "down" direction for the end effector and the "forward" direction for the end effector).

        Args:
            u: the "down" direction for the end effector.
            v: the "forward" direction for the end effector.
        """
        return calc_ee_quat_from_directions(u, v, self.get_ee_default_quat())

    def get_gripper_state(self) -> Optional[bool]:
        """Get the gripper state.

        Returns:
            Optional[bool]: True if the gripper is activated.
        """
        raise NotImplementedError()

    def fk(self, qpos: np.ndarray, link_name_or_id: Optional[Union[str, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Forward kinematics.

        Args:
            qpos: Joint configuration.
            link_name_or_id: name or id of the link. If not specified, the pose of the end effector is returned.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 3D position and 4D quaternion of the end effector.
        """
        robot_id = self.get_body_id()
        with BodyFullStateSaver(self.client.w, robot_id):
            self.set_qpos(qpos)
            if link_name_or_id is None:
                return self.get_ee_pose(fk=True)
            elif isinstance(link_name_or_id, str):
                state = self.client.w.get_link_state(link_name_or_id, fk=True)
                return np.array(state.position), np.array(state.orientation)
            elif isinstance(link_name_or_id, int):
                state = self.client.w.get_link_state_by_id(robot_id, link_name_or_id, fk=True)
                return np.array(state.position), np.array(state.orientation)
            else:
                raise TypeError(f'link_name_or_id must be str or int, got {type(link_name_or_id)}.')

    def ik(self, pos: np.ndarray, quat: np.ndarray, force: bool = False, max_distance: float = float('inf'), max_attempts: int = 1000, verbose: bool = False) -> Optional[np.ndarray]:
        """Inverse kinematics.

        Args:
            pos: 3D position of the end effector.
            quat: 4D quaternion of the end effector.
            force: Whether to force the IK solver to return a solution. Defaults to False.
                If set, the IK solve may return a solution even if the end effector is not at the given pose.
                This function is useful for debugging and for moving towards a certain direction.
            max_distance: Maximum distance between the last qpos and the solution (Only used for IKFast). Defaults to float('inf').
            max_attempts: Maximum number of attempts (only used for IKFast). Defaults to 1000.
            verbose: Whether to print debug information.

        Returns:
            np.ndarray: Joint configuration.
        """

        if self.ik_method is IKMethod.PYBULLET:
            assert max_distance == float('inf'), 'max_distance is not supported in PyBullet IK'
            return self.ik_pybullet(pos, quat, force=force, verbose=verbose)
        elif self.ik_method is IKMethod.IKFAST:
            assert force is False, 'force is not supported in ik_fast'
            return self.ikfast(pos, quat, max_distance=max_distance, max_attempts=max_attempts, verbose=verbose)

    def ik_pybullet(self, pos: np.ndarray, quat: np.ndarray, force: bool = False, pos_tol: float = 1e-2, quat_tol: float = 1e-2, verbose: bool = False) -> Optional[np.ndarray]:
        """Inverse kinematics using pybullet.

        Args:
            pos: 3D position of the end effector.
            quat: 4D quaternion of the end effector.
            force: Whether to force the IK solver to return a solution. Defaults to False.
                If set, the IK solve may return a solution even if the end effector is not at the given pose.
                This function is useful for debugging and for moving towards a certain direction.
            pos_tol: tolerance of the position. Defaults to 1e-2.
            quat_tol: tolerance of the quaternion. Defaults to 1e-2.
            verbose: Whether to print debug information.

        Returns:
            np.ndarray: Joint configuration.
        """
        lower_limits, upper_limits = self.get_full_joint_limits()
        joints = self.client.p.calculateInverseKinematics(
            bodyUniqueId=self.get_body_id(),
            endEffectorLinkIndex=self.get_ee_link_id(),
            targetPosition=pos,
            targetOrientation=quat,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=upper_limits - lower_limits,
            restPoses=self.get_full_home_qpos(),
            maxNumIterations=100,
            residualThreshold=1e-5,
        )

        if verbose:
            print('IK (solution, lower, upper):\n', np.stack([joints[:7], lower_limits, upper_limits], axis=0), sep='')

        joints = np.array(joints)[:self.get_dof()]
        lower_limits = lower_limits[:self.get_dof()]
        upper_limits = upper_limits[:self.get_dof()]

        if np.all(lower_limits <= joints) and np.all(joints <= upper_limits):
            qpos = np.array(joints[:7])
            fk_pos, fk_quat = self.fk(qpos)
            if np.linalg.norm(fk_pos - pos) < pos_tol and 1 - np.abs(quat_mul(quat, quat_conjugate(fk_quat))[3]) < quat_tol:
                return qpos
            else:
                if verbose:
                    print('IK failed: pos error:', np.linalg.norm(fk_pos - pos), 'quat error:', 1 - np.abs(quat_mul(quat, quat_conjugate(fk_quat))[3]))

        if force:
            return np.array(joints[:7])

        return None

    def ikfast(self, pos: np.ndarray, quat: np.ndarray, last_qpos: Optional[np.ndarray] = None, max_attempts: int = 1000, max_distance: float = float('inf'), error_on_fail: bool = True, verbose: bool = False) -> Optional[np.ndarray]:
        """Inverse kinematics using IKFast.

        Args:
            pos: 3D position of the end effector.
            quat: 4D quaternion of the end effector.
            last_qpos: Last joint configuration. Defaults to None.
                If None, the current joint configuration is used.
            max_attempts: Maximum number of IKFast attempts. Defaults to 1000.
            max_distance: Maximum distance between the target pose and the end effector. Defaults to float('inf').
            error_on_fail: Whether to raise an error if the IKFast solver fails. Defaults to True.
            verbose: Whether to print debug information.

        Returns:
            np.ndarray: Joint configuration.
        """
        raise NotImplementedError(f'IKFast is not supported for this {self.__class__.__name__}.')

    def get_mass_matrix(self, qpos: Optional[np.ndarray] = None) -> np.ndarray:
        """Get the mass matrix.

        Args:
            qpos: Joint configuration.

        Returns:
            np.ndarray: Mass matrix.
        """
        if qpos is None:
            qpos = self.get_qpos()

        if self.get_dof() < self.get_full_dof():
            qpos = np.append(qpos, [0.0 for _ in range(self.get_full_dof() - self.get_dof())])

        mass_matrix = self.client.p.calculateMassMatrix(self.get_body_id(), qpos.tolist())
        mass_matrix = np.array(mass_matrix)
        if mass_matrix.shape[0] > self.get_dof():
            mass_matrix = mass_matrix[:self.get_dof(), :self.get_dof()]
        return mass_matrix

    def get_jacobian(self, qpos: Optional[np.ndarray] = None, link_id: Optional[int] = None) -> np.ndarray:
        """Get the Jacobian matrix.

        Args:
            qpos: Joint configuration.
            link_id: id of the link. If not specified, the Jacobian of the end effector is returned.

        Returns:
            np.ndarray: Jacobian matrix. The shape is (6, nr_moveable_joints).
        """
        if link_id is None:
            link_id = self.get_ee_link_id()

        if qpos is None:
            qpos = self.get_qpos()

        if self.get_dof() < self.get_full_dof():
            qpos = np.append(qpos, [0.0 for _ in range(self.get_full_dof() - self.get_dof())])

        linear_jacobian, angular_jacobian = self.client.p.calculateJacobian(
            bodyUniqueId=self.get_body_id(), linkIndex=link_id,
            localPosition=[0.0, 0.0, 0.0],
            objPositions=qpos.tolist(), objVelocities=np.zeros_like(qpos).tolist(), objAccelerations=np.zeros_like(qpos).tolist()
        )

        jacobian = np.vstack([np.array(linear_jacobian), np.array(angular_jacobian)])
        if jacobian.shape[1] > self.get_dof():
            jacobian = jacobian[:, :self.get_dof()]
        return jacobian

    def get_coriolis_torque(self, qpos: Optional[np.ndarray] = None, qvel: Optional[np.ndarray] = None) -> np.ndarray:
        """Get the Coriolis torque.

        Args:
            qpos: Joint configuration.
            qvel: Joint velocity.

        Returns:
            np.ndarray: Coriolis torque.
        """
        if qpos is None:
            qpos = self.get_full_qpos()
        if qvel is None:
            qvel = self.get_full_qvel()

        qddot = self.client.p.calculateInverseDynamics(self.get_body_id(), qpos.tolist(), qvel.tolist(), np.zeros_like(qvel).tolist())
        qddot = np.array(qddot)
        if qddot.shape[0] > self.get_dof():
            qddot = qddot[:self.get_dof()]
        return qddot

    def get_configuration_space(self) -> BoxConfigurationSpace:
        if self._cspace is None:
            ranges = list()
            joint_limits = self.get_joint_limits()
            for lower, upper in zip(joint_limits[0], joint_limits[1]):
                ranges.append(Range(lower, upper))
            self._cspace = BoxConfigurationSpace(ranges, 0.02)

        return self._cspace

    def get_collision_free_problem_space(self, ignored_collision_bodies: Optional[Sequence[int]] = None) -> CollisionFreeProblemSpace:
        if self._cfree_default_pspace is not None and ignored_collision_bodies is None:
            return self._cfree_default_pspace

        cspace = self.get_configuration_space()

        def is_colliding(q):
            return self.is_colliding(q, ignored_collision_bodies=ignored_collision_bodies)

        pspace = CollisionFreeProblemSpace(cspace, is_colliding)
        if ignored_collision_bodies is not None:
            self._cfree_default_pspace = pspace
        return pspace

    def is_colliding_with_saved_state(self, q: Optional[np.ndarray] = None, return_contacts: bool = False, ignored_collision_bodies: Optional[Sequence[int]] = None):
        if q is not None:
            with self.world.save_world_builtin():
                return self.is_colliding(q, return_contacts, ignored_collision_bodies)

        # If q is None, we don't need to save the world state.
        return self.is_colliding(q, return_contacts, ignored_collision_bodies)

    def is_colliding(self, q: Optional[np.ndarray] = None, return_contacts: bool = False, ignored_collision_bodies: Optional[Sequence[int]] = None):
        """Check if the robot is colliding with other objects. When the joint configuration (q) is provided, we will set the robot to that configuration before checking the collision.
        Note that this function will not restore the robot to the original configuration after the check. If you want to restore the robot to the original configuration,
        you should use :meth:`is_colliding_with_saved_state` instead.

        Args:
            q: Joint configuration. If None, the current joint configuration is used.
            return_contacts: whether to return the contact information. Defaults to False.
            ignored_collision_bodies: IDs of the objects to be ignored in collision checking. Defaults to None.
        """
        raise NotImplementedError()

    def rrt_collision_free(self, qpos1: np.ndarray, qpos2: Optional[np.ndarray] = None, ignored_collision_bodies: Optional[Sequence[int]] = None, smooth_fine_path: bool = False, disable_renderer: bool = True, **kwargs):
        """RRT-based collision-free path planning.

        Args:
            qpos1: Start position.
            qpos2: End position. If None, the current position is used.
            ignored_collision_bodies: IDs of the objects to ignore in collision checking. Defaults to None.
            smooth_fine_path: Whether to smooth the path. Defaults to False.
            disable_renderer: Whether to disable the renderer. Defaults to True.
            kwargs: Additional arguments.

        Returns:
            bool: True if the path is collision-free.
            List[np.ndarray]: Joint configuration trajectory.
        """
        if qpos2 is None:
            qpos2 = qpos1
            qpos1 = self.get_qpos()

        cfree_pspace = self.get_collision_free_problem_space(ignored_collision_bodies=ignored_collision_bodies)
        with self.world.save_world_builtin(), cond_with(self.client.disable_rendering(suppress_stdout=False), disable_renderer):
            path = birrt(cfree_pspace, qpos1, qpos2, smooth_fine_path=smooth_fine_path, **kwargs)
            if path[0] is not None:
                return True, path[0]
            return False, None

    def set_arm_joint_position_control(self, target_qpos: np.ndarray, control_mode: int = pybullet.POSITION_CONTROL, gains: float = 0.3, set_gripper_control: bool = True):
        """Set the arm joint position control.

        Args:
            target_qpos: target joint configuration.
            control_mode: control mode.
            gains: gains of the controller. Defaults to 0.3.
            set_gripper_control: whether to set the gripper control. Defaults to True.
        """
        raise NotImplementedError()

    def set_ee_impedance_control(
        self, target_pos: np.ndarray, target_quat: np.ndarray,
        kp_pos: Union[float, np.ndarray] = 200, kp_ori: Union[float, np.ndarray] = 1,
        kd_pos: Optional[Union[float, np.ndarray]] = None, kd_ori: Optional[Union[float, np.ndarray]] = 0.01, max_torque: float = 100,
        damping_scale: float = 2.0,
        simulate_with_position_pd: bool = False,
        tau_to_qpos_ratio: float = 0.0005,
        set_gripper_control: bool = True,
        verbose: bool = True,
    ):
        """Set the end-effector impedance control command.

        Args:
            target_pos: the target position of the end-effector.
            target_quat: the target orientation of the end-effector.
            kp_pos: the proportional gains for the position control.
            kp_ori: the proportional gains for the orientation control.
            kd_pos: the derivative gains for the position control.
            kd_ori: the derivative gains for the orientation control. It is recommended to manual set this to a small value to avoid oscillation (instead of using the critical damping).
            max_torque: the maximum torque allowed for the robot.
            damping_scale: the scale for the damping term (as a multiplicative term to the critical damping).
            simulate_with_position_pd: if True, we will simulate the torque control using the position control. This is useful for debugging and it can be more stable.
            tau_to_qpos_ratio: the ratio between the torque and the joint position. This is only used when `simulate_with_position_pd` is True. We will use `qpos += tau * tau_to_qpos_ratio` to simulate the position control.
            set_gripper_control: if True, the gripper control will be set.
            verbose: if True, the debug information will be printed.
        """
        raise NotImplementedError()

    def set_full_hold_position_control(self, qpos: Optional[np.ndarray] = None, gains: float = 0.3):
        """Set the control parameter for holding the current arm joint positions, while the gripper holding the object."""
        qpos = self.get_qpos() if qpos is None else qpos
        self.set_arm_joint_position_control(qpos, gains=gains, set_gripper_control=True)

    def move_qpos(self, target_qpos: np.ndarray, speed: float = 0.01, timeout: float = 10.0, local_smoothing: bool = True) -> bool:
        """Move the robot to the given joint configuration.

        Args:
            target_qpos: Target joint configuration.
            speed: Speed of the movement. Defaults to 0.01.
            timeout: Timeout of the movement. Defaults to 10.
            local_smoothing: Whether to use local smoothing. Defaults to True.

        Returns:
            bool: True if the movement is successful.
        """
        raise NotImplementedError()

    def move_home(self, timeout: float = 10.0) -> bool:
        """Move the robot to the home configuration."""
        return self.move_qpos(self.get_home_qpos(), timeout=timeout)

    def move_home_cfree(self, speed: float = 0.01, timeout: float = 10.0):
        with self.client.disable_rendering(suppress_stdout=False):
            try:
                success, qpath = self.rrt_collision_free(self.get_home_qpos())
            except ValueError:
                success = False

        if not success:
            if not self.warnings_suppressed:
                logger.warning('Cannot find a collision-free path to home. Doing a direct move.')
            return self.move_qpos(self.get_home_qpos(), speed=speed, timeout=timeout)
        return self.move_qpos_path(qpath, speed=speed, timeout=timeout)

    def move_pose(self, pos, quat, speed=0.01, force: bool = False, verbose: bool = False) -> bool:
        """Move the end effector to the given pose.

        Args:
            pos: 3D position of the end effector.
            quat: 4D quaternion of the end effector.
            speed: Speed of the movement. Defaults to 0.01.
            force: Whether to force the IK solver to return a solution. Defaults to False.
                If set, the IK solve may return a solution even if the end effector is not at the specified pose.
            verbose: Whether to print debug information.

        Returns:
            bool: True if the movement is successful.
        """

        if verbose:
            logger.info(f'{self.body_name}: Moving to pose: {pos}, {quat}.')
        target_qpos = self.ik(pos, quat, force=force)
        rv = self.move_qpos(target_qpos, speed)
        return rv

    def move_pose_smooth(self, pos: np.ndarray, quat: np.ndarray, speed: float = 0.01, max_qpos_distance: float = 1.0) -> bool:
        if self.ik_method is IKMethod.IKFAST:
            # In ik_fast mode, we need to specify the `max_qpos_distance` to avoid the IK solution to be too far away.
            target_qpos = self.ikfast(pos, quat, max_distance=max_qpos_distance)
        else:
            target_qpos = self.ik_pybullet(pos, quat)

        rv = self.move_qpos(target_qpos, speed=speed)
        return rv

    def move_qpos_path(self, qpos_trajectory: Iterable[np.ndarray], speed: float = 0.01, timeout: float = 10, first_timeout: float = None, local_smoothing: bool = True, verbose: bool = False) -> bool:
        """Move the robot along the given joint configuration trajectory.

        Args:
            qpos_trajectory: Joint configuration trajectory.
            speed: Speed of the movement. Defaults to 0.01.
            timeout: Timeout of the movement. Defaults to 10.
            first_timeout: Timeout of the first movement. Defaults to None (same as timeout).
            local_smoothing: Whether to use local smoothing. Defaults to True.
            verbose: Whether to print debug information.

        Returns:
            bool: True if the movement is successful.
        """

        this_timeout = timeout if first_timeout is None else first_timeout
        rv = True
        for target_qpos in qpos_trajectory:
            if verbose:
                logger.info(f'{self.body_name}: Moving to qpos: {target_qpos}.')
            rv = self.move_qpos(target_qpos, speed, this_timeout, local_smoothing=local_smoothing)
            this_timeout = timeout
            # if verbose:
            #     logger.info(f'{self.body_name}: Moved to qpos:  {self.get_qpos()}.')
        if not rv:
            return False
        return True

    def move_qpos_path_v2_set_control(
        self, qpos_trajectory: Iterable[np.ndarray],
        step_size: float = 1, gains: float = 0.3,
        atol: float = 0.03, timeout: float = 20,
        verbose: bool = False,
        return_world_states: bool = False,
    ):
        raise NotImplementedError()

    def move_qpos_path_v2(self, qpos_trajectory: Iterable[np.ndarray], timeout: float = 20) -> bool:
        """A customized version of move_qpos_trajectory. It allows the user to specify path tracking mechanisms.

        Args:
            qpos_trajectory: joint configuration trajectory.
            timeout: timeout of the movement.

        Returns:
            bool: True if the movement is successful.
        """
        raise NotImplementedError()

    def replay_qpos_trajectory(self, qpos_trajectory: Iterable[np.ndarray], verbose: bool = True):
        """Replay a joint confifguartion trajectory. This function is useful for debugging a generated joint trajectory.

        Args:
            qpos_trajectory: joint configuration trajectory.
            verbose: whether to print debug information.
        """
        for i, target_qpos in enumerate(qpos_trajectory):
            if verbose:
                logger.info(f'{self.body_name}: Moving to qpos: {target_qpos}. Trajectory replay {i + 1}/{len(qpos_trajectory)}.')
            self.set_qpos_with_holding(target_qpos)
            self.client.wait_for_duration(0.1)
            # input('Press enter to continue.')

    def create_gripper_constraint(self, object_id, verbose: bool = False):
        if self.gripper_constraint is not None:
            self.client.p.removeConstraint(self.gripper_constraint)
            self.gripper_constraint = None

        if verbose:
            logger.info(f'{self.body_name}: Creating constraint between {self.get_body_id()} and {object_id}.')

        ee_link = self.get_ee_link_id()

        ee_pose = self.world.get_link_state_by_id(self.get_body_id(), ee_link, fk=True)
        obj_pose = self.p.getBasePositionAndOrientation(object_id)
        ee_to_world = pybullet.invertTransform(ee_pose[0], ee_pose[1])
        ee_to_object = pybullet.multiplyTransforms(ee_to_world[0], ee_to_world[1], obj_pose[0], obj_pose[1])
        self.gripper_constraint = self.p.createConstraint(
            parentBodyUniqueId=self.get_body_id(),
            parentLinkIndex=ee_link,
            childBodyUniqueId=object_id,
            childLinkIndex=-1,  # should be the link index of the contact link.
            jointType=pybullet.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=ee_to_object[0],
            parentFrameOrientation=ee_to_object[1],
            childFramePosition=(0, 0, 0),
            childFrameOrientation=(0, 0, 0)
        )

    def set_qpos_with_holding(self, qpos: np.ndarray) -> None:
        self.set_qpos(qpos)
        if self.gripper_constraint is not None:
            # TODO(Jiayuan Mao @ 2023/03/02): should consider the actual link that the gripper is attached to, and self-collision, etc.
            constraint = self.p.getConstraintInfo(self.gripper_constraint)
            other_body_id = constraint[2]
            parent_frame_pos = constraint[6]
            parent_frame_orn = constraint[8]

            ee_link = self.get_ee_link_id()
            ee_pose = self.world.get_link_state_by_id(self.get_body_id(), ee_link, fk=True)
            ee_to_object = (parent_frame_pos, parent_frame_orn)
            obj_pose = pybullet.multiplyTransforms(ee_pose[0], ee_pose[1], *ee_to_object)
            self.w.set_body_state2_by_id(other_body_id, obj_pose[0], obj_pose[1])

    def attach_object(self, object_id: int, ee_to_object: Tuple[np.ndarray, np.ndarray]) -> None:
        assert self.use_magic_gripper, 'The magic gripper is not enabled.'
        assert self.gripper_constraint is None, 'The gripper is already attached to an object.'

        object_to_ee = pybullet.invertTransform(ee_to_object[0], ee_to_object[1])
        world_to_ee = self.world.get_link_state_by_id(self.get_body_id(), self.get_ee_link_id(), fk=True)
        world_to_object = pybullet.multiplyTransforms(world_to_ee[0], world_to_ee[1], *object_to_ee)
        self.w.set_body_state2_by_id(object_id, world_to_object[0], world_to_object[1])

        # NB(Jiayuan Mao @ 2024/08/2): We will first create a constraint between the gripper and the object --- this ensures that the object will stay the same when the gripper moves.
        self.create_gripper_constraint(object_id)

    def detach_object(self) -> None:
        assert self.use_magic_gripper, 'The magic gripper is not enabled.'
        assert self.gripper_constraint is not None, 'The gripper is not attached to any object.'

        if self.gripper_constraint is not None:
            self.client.p.removeConstraint(self.gripper_constraint)
            self.gripper_constraint = None

    def get_attached_object(self) -> Optional[int]:
        if self.gripper_constraint is not None:
            constraint = self.p.getConstraintInfo(self.gripper_constraint)
            return constraint[2]
        return None

    def get_attached_object_pose_in_ee_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.gripper_constraint is not None:
            constraint = self.p.getConstraintInfo(self.gripper_constraint)
            parent_frame_pos = constraint[6]
            parent_frame_orn = constraint[8]
            return parent_frame_pos, parent_frame_orn
        return None


class BulletRobotActionPrimitive(object):
    def __init__(self, robot: BulletArmRobotBase):
        self.robot = robot

    def __call__(self, *args, **kwargs) -> bool:
        raise NotImplementedError()

    @property
    def client(self):
        return self.robot.client

    @property
    def p(self):
        return self.robot.client.p

    @property
    def world(self):
        return self.robot.client.w

    @property
    def w(self):
        return self.robot.client.w

    @property
    def warnings_suppressed(self):
        return self.robot.warnings_suppressed
