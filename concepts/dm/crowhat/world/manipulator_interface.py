#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : manipulator_interface.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/23/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Union, Iterator, Tuple, List, Dict, Callable

import jacinle
from concepts.algorithm.configuration_space import BoxConfigurationSpace, CollisionFreeProblemSpace
from concepts.algorithm.rrt.rrt import birrt
from concepts.dm.crow.interfaces.controller_interface import CrowControllerExecutionError
from concepts.dm.crowhat.manipulation_utils.pose_utils import canonicalize_pose, pose_difference, pose_distance2
from concepts.dm.crowhat.world.planning_world_interface import AttachmentInfo, PlanningWorldInterface
from concepts.math.range import Range
from concepts.math.frame_utils_xyzw import get_transform_a_to_b, solve_ee_from_tool, calc_ee_quat_from_directions
from concepts.utils.typing_utils import VecNf, Vec3f, Vec4f

__all__ = [
    'SingleGroupMotionPlanningInterface', 'MotionPlanningResult', 'MotionPlanningContactTimeCollisionCheckingArguments',
    'RobotControllerExecutionFailed', 'RobotControllerExecutionContext', 'RobotArmJointTrajectory',
    'SingleArmControllerInterface'
]


class MotionPlanningResult(object):
    def __init__(self, success: bool, result: Any, error: str = ''):
        self.succeeded = success
        self.result = result
        self.error = error

    def __bool__(self):
        return self.succeeded

    def __str__(self):
        if self.succeeded:
            return f'MotionPlanningResult(SUCC: {self.result})'
        return f'MotionPlanningResult(FAIL: error="{self.error}")'

    def __repr__(self):
        return str(self)

    @classmethod
    def success(cls, result: Any):
        return cls(True, result)

    @classmethod
    def fail(cls, error: str):
        return cls(False, None, error)


@dataclass
class MotionPlanningContactTimeCollisionCheckingArguments(object):
    qpos_range: float
    collision_tol: float


class SingleGroupMotionPlanningInterface(object):
    """General interface for robot arms. It specifies a set of basic operations for motion planning:

    - ``ik``: inverse kinematics.
    - ``fk``: forward kinematics.
    - ``jac``: jacobian matrix.
    - ``coriolis``: coriolis torque.
    - ``mass``: mass matrix.
    """

    def __init__(self, planning_world: Optional[PlanningWorldInterface] = None):
        self._planning_world = planning_world
        self._rrt_collision_distance = 0.01
        self._configuration_space_max_stepdiff = 0.02
        self._configuration_space_extra_validation_func = None

    @property
    def planning_world(self) -> Optional[PlanningWorldInterface]:
        return self._planning_world

    def set_planning_world(self, planning_world: Optional[PlanningWorldInterface]):
        self._planning_world = planning_world

    def set_rrt_collision_distance(self, distance: Optional[float]):
        self._rrt_collision_distance = distance

    def get_default_configuration_space_max_stepdiff(self) -> float:
        """Get the default maximum step difference for the configuration space. Default is 0.02 rad."""
        # return np.pi / 180 * 5
        return self._configuration_space_max_stepdiff

    def set_default_configuration_space_max_stepdiff(self, max_stepdiff: float):
        self._configuration_space_max_stepdiff = max_stepdiff

    def set_configuration_space_extra_validation_func(self, extra_validation_func: Optional[Callable[[np.ndarray], bool]]):
        self._configuration_space_extra_validation_func = extra_validation_func

    def get_configuration_space(self, max_stepdiff: Optional[float] = None) -> BoxConfigurationSpace:
        if max_stepdiff is None:
            max_stepdiff = self.get_default_configuration_space_max_stepdiff()

        lower, upper = self.joint_limits
        return BoxConfigurationSpace(
            [Range(l, u) for l, u in zip(lower, upper)],
            cspace_max_stepdiff=max_stepdiff,
            extra_validation_func=self._configuration_space_extra_validation_func
        )

    def get_collision_free_problem_space(
        self,
        ignored_collision_bodies: Optional[List[Union[str, int]]] = None,
        max_stepdiff: Optional[float] = None,
        collision_tol: Optional[float] = None,
        move_from_contact_collision_args: Optional[MotionPlanningContactTimeCollisionCheckingArguments] = None,
        move_to_contact_collision_args: Optional[MotionPlanningContactTimeCollisionCheckingArguments] = None,
        move_from_qpos: Optional[VecNf] = None,
        move_to_qpos: Optional[VecNf] = None
    ) -> CollisionFreeProblemSpace:
        jacinle.lf_indent_print(f'Get collision free problem space move_from_contact_collision_args: {move_from_contact_collision_args} move_to_contact_collision_args: {move_to_contact_collision_args}')
        # Initialize the CollisionChecker
        collision_checker = CollisionChecker(
            self.check_collision,
            ignored_collision_bodies=ignored_collision_bodies,
            collision_tol=collision_tol,
            move_from_contact_collision_args=move_from_contact_collision_args,
            move_to_contact_collision_args=move_to_contact_collision_args,
            move_from_qpos=move_from_qpos,
            move_to_qpos=move_to_qpos,
        )

        # Use the collision_checker instance as the collide_fn
        return CollisionFreeProblemSpace(self.get_configuration_space(max_stepdiff=max_stepdiff), collision_checker)

    def check_collision(
        self, qpos: Optional[VecNf] = None, ignore_self_collision: bool = True,
        ignored_collision_bodies: Optional[List[Union[str, int]]] = None,
        max_distance: Optional[float] = None,
        checkpoint_world_state: Optional[bool] = None,
        return_list: bool = False, verbose: bool = False
    ) -> Union[bool, List[Union[str, int]]]:
        """Check if the current configuration is in collision.

        Args:
            qpos: the configuration to check. If None, it will use the current configuration.
            ignore_self_collision: whether to ignore the collision between the robot and itself (including the attached objects).
            ignored_collision_bodies: a list of identifiers of the bodies to ignore.
            checkpoint_world_state: whether to checkpoint the world state before checking collision. If None, we will checkpoint the world state if qpos is not None.
            return_list: whether to return the list of collision bodies.

        Returns:
            True if the configuration is in collision, False otherwise.
        """
        if checkpoint_world_state is None:
            checkpoint_world_state = qpos is not None

        backup_qpos = None
        if checkpoint_world_state:
            backup_qpos = self.get_qpos().copy()

        if qpos is not None:
            self.set_qpos(qpos)

        if ignored_collision_bodies is None:
            ignored_collision_bodies = list()

        if ignore_self_collision:
            if (attached := self.get_attached_objects()) is not None:
                for attachment in attached:
                    ignored_collision_bodies.append(attachment.body_b)
            ignored_collision_bodies.append(self.body_id)

        try:
            rv = self.planning_world.check_collision_with_other_objects(
                self.body_id, ignore_self_collision=ignore_self_collision, ignored_collision_bodies=ignored_collision_bodies, max_distance=max_distance,
                return_list=return_list, verbose=verbose
            )
            if rv:
                return True

            # TODO(Jiayuan Mao @ 2024/12/20): Fix the attached object collision checking.
            # In theory, we should only ignore the collision between the end effector and the attached object, but keep the collision checking between
            # the attached object and the rest part of the robot body.
            if (attached_objects := self.get_attached_objects()) is not None:
                for attachment in attached_objects:
                    if attachment.body_b in ignored_collision_bodies:
                        ignored_collision_bodies.remove(attachment.body_b)
                for attachment in attached_objects:
                    rv = self.planning_world.check_collision_with_other_objects(
                        attachment.body_b, ignore_self_collision=ignore_self_collision, ignored_collision_bodies=ignored_collision_bodies, max_distance=max_distance,
                        return_list=return_list, verbose=verbose
                    )
                    if rv:
                        return True

            return False
        finally:
            if checkpoint_world_state:
                self.set_qpos(backup_qpos)

    def rrt_collision_free(
        self, qpos1: np.ndarray, qpos2: Optional[np.ndarray] = None,
        ignored_collision_bodies: Optional[List[Union[str, int]]] = None, smooth_fine_path: bool = False,
        move_from_contact_collision_args: Optional[MotionPlanningContactTimeCollisionCheckingArguments] = None,
        move_to_contact_collision_args: Optional[MotionPlanningContactTimeCollisionCheckingArguments] = None,
        verbose: bool = False,
        **inner_rrt_kwargs
    ) -> Tuple[bool, Optional[List[np.ndarray]]]:
        """RRT-based collision-free path planning.

        Args:
            qpos1: Start position.
            qpos2: End position. If None, the current position is used.
            ignored_collision_bodies: IDs of the objects to ignore in collision checking. Defaults to None.
            smooth_fine_path: Whether to smooth the path. Defaults to False.
            move_from_contact_collision_args: Collision checking arguments for the start position. Defaults to None. Useful when the starting position is in contact with the environment.
            move_to_contact_collision_args: Collision checking arguments for the end position. Defaults to None. Useful when the ending position is in contact with the environment.
            kwargs: Additional arguments.

        Returns:
            bool: True if the path is collision-free.
            List[np.ndarray]: Joint configuration trajectory.
        """
        if qpos2 is None:
            qpos2 = qpos1
            qpos1 = self.get_qpos()

        cfree_pspace = self.get_collision_free_problem_space(
            ignored_collision_bodies=ignored_collision_bodies, collision_tol=self._rrt_collision_distance,
            move_from_contact_collision_args=move_from_contact_collision_args,
            move_to_contact_collision_args=move_to_contact_collision_args,
            move_from_qpos=qpos1, move_to_qpos=qpos2
        )
        with self.planning_world.checkpoint_world():
            path = birrt(cfree_pspace, qpos1, qpos2, smooth_fine_path=smooth_fine_path, verbose=verbose, **inner_rrt_kwargs)
            if path[0] is not None:
                return True, path[0]
            return False, None

    @property
    def nr_joints(self) -> int:
        return self.get_nr_joints()

    def get_nr_joints(self) -> int:
        raise NotImplementedError()

    @property
    def joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        lower, upper = self.get_joint_limits()
        return np.asarray(lower), np.asarray(upper)

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    @property
    def body_id(self) -> int:
        return self.get_body_id()

    def get_body_id(self) -> int:
        raise NotImplementedError()

    @property
    def ee_link_id(self) -> int:
        return self.get_ee_link_id()

    def get_ee_link_id(self) -> int:
        raise NotImplementedError()

    @property
    def ee_default_quat(self) -> Vec4f:
        return self.get_ee_default_quat()

    def get_ee_default_quat(self) -> Vec4f:
        """Get the default end-effector quaternion. This is defined by the quaternion of the EE when

            1. the robot base is at the origin,
            2. the "down" direction is (0, 0, -1), and
            3. the "forward" direction is (1, 0, 0).

        Returns:
            Vec4f: The default end-effector quaternion.
        """
        raise NotImplementedError()

    def get_qpos(self) -> np.ndarray:
        return np.asarray(self._get_qpos())

    def _get_qpos(self) -> VecNf:
        raise NotImplementedError()

    def set_qpos(self, qpos: VecNf):
        self._set_qpos(np.asarray(qpos))

    def _set_qpos(self, qpos: np.ndarray):
        raise NotImplementedError()

    def get_ee_pose(self) -> Tuple[Vec3f, Vec4f]:
        return self._planning_world.get_link_pose(self.body_id, self.ee_link_id)

    def get_attached_objects(self) -> List[AttachmentInfo]:
        return self._get_attached_objects()

    def _get_attached_objects(self) -> List[AttachmentInfo]:
        raise NotImplementedError()

    def add_attachment(self, body: Union[str, int], link: Optional[int] = None, self_link: Optional[int] = None, ee_to_object: Optional[Tuple[Vec3f, Vec4f]] = None) -> Any:
        return self._add_attachment(body, link, self_link, ee_to_object)

    def _add_attachment(self, body: Union[str, int], link: Optional[int] = None, self_link: Optional[int] = None, ee_to_object: Optional[Tuple[Vec3f, Vec4f]] = None) -> Any:
        raise NotImplementedError()

    def remove_attachment(self, body: Union[str, int, None], link: Optional[int] = None, self_link: Optional[int] = None):
        self._remove_attachment(body, link, self_link)

    def _remove_attachment(self, body: Union[str, int, None], link: Optional[int] = None, self_link: Optional[int] = None):
        raise NotImplementedError()

    def fk(self, qpos: VecNf) -> Tuple[Vec3f, Vec4f]:
        return self._fk(np.asarray(qpos))

    def _fk(self, qpos: np.ndarray) -> Tuple[Vec3f, Vec4f]:
        raise NotImplementedError()

    def ik(self, pos: Union[Vec3f, Tuple[Vec3f, Vec4f]], quat: Optional[Vec4f] = None, qpos: Optional[VecNf] = None, max_distance: Optional[float] = None) -> Optional[np.ndarray]:
        pos, quat = canonicalize_pose(pos, quat)
        return self._ik(pos, quat, qpos, max_distance=max_distance)

    def _ik(self, pos: np.ndarray, quat: np.ndarray, qpos: Optional[np.ndarray] = None, max_distance: Optional[float] = None) -> Optional[np.ndarray]:
        raise NotImplementedError()

    def gen_ik(self, pos: Union[Vec3f, Tuple[Vec3f, Vec4f]], quat: Optional[Vec4f] = None, qpos: Optional[VecNf] = None, max_distance: Optional[float] = None, max_returns: int = 5) -> Iterator[np.ndarray]:
        pos, quat = canonicalize_pose(pos, quat)
        for i in range(max_returns):
            rv = self._ik(pos, quat, qpos, max_distance=max_distance)
            if rv is not None:
                yield rv

    def jacobian(self, qpos: VecNf) -> np.ndarray:
        return self._jacobian(np.asarray(qpos))

    def _jacobian(self, qpos: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def coriolis(self, qpos: VecNf, qvel: VecNf) -> np.ndarray:
        return self._coriolis(np.asarray(qpos), np.asarray(qvel))

    def _coriolis(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def mass(self, qpos: VecNf) -> np.ndarray:
        return self._mass(np.asarray(qpos))

    def _mass(self, qpos: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def calc_differential_ik_qpos_diff(self, current_qpos: VecNf, current_pose: Tuple[Vec3f, Vec4f], next_pose: Tuple[Vec3f, Vec4f]) -> np.ndarray:
        """Use the differential IK to compute the joint difference for the given pose difference."""
        current_pose = canonicalize_pose(current_pose)
        next_pose = canonicalize_pose(next_pose)
        J = self.jacobian(current_qpos)  # 6 x N

        solution = np.linalg.pinv(J) @ pose_difference(current_pose, next_pose)
        # solution = np.linalg.lstsq(J, pose_difference(current_pose, next_pose), rcond=None)[0]
        return solution

    def calc_differential_ik(self, current_qpos: VecNf, current_pose: Tuple[Vec3f, Vec4f], next_pose: Tuple[Vec3f, Vec4f], max_iterations: int = 10, pos_error_tol: float = 1e-3, rot_error_tol: float = 1e-3) -> np.ndarray:
        for i in range(max_iterations):
            if i > 0:
                current_pose = self.fk(current_qpos)

            pos_error, rot_error = pose_distance2(current_pose, next_pose)
            if pos_error < pos_error_tol and rot_error < rot_error_tol:
                break

            qpos = np.asarray(current_qpos) + self.calc_differential_ik_qpos_diff(current_qpos, current_pose, next_pose)
            lower, upper = self.joint_limits
            current_qpos = np.clip(qpos, lower, upper)

        return current_qpos

    def calc_ee_pose_from_single_attached_object_pose(self, pos: Vec3f, quat: Vec4f) -> Tuple[Vec3f, Vec4f]:
        """Get the end-effector pose given the desired pose of the attached object."""
        attached_object = self.get_attached_objects()
        if len(attached_object) != 1:
            raise ValueError('The end-effector should have exactly one attached object.')
        attached_object = attached_object[0].body_b

        world_to_ee = self.get_ee_pose()
        world_to_obj = self._planning_world.get_object_pose(attached_object)
        ee_to_tool = get_transform_a_to_b(world_to_ee[0], world_to_ee[1], world_to_obj[0], world_to_obj[1])

        return solve_ee_from_tool(pos, quat, ee_to_tool)

    def calc_ee_quat_from_vectors(self, u: Vec3f = (-1., 0., 0.), v: Vec3f = (1., 0., 0.)) -> Vec4f:
        """Compute the quaternion from two directions (the "down" direction for the end effector and the "forward" direction for the end effector).

        Args:
            u: the "down" direction for the end effector.
            v: the "forward" direction for the end effector.
        """
        return calc_ee_quat_from_directions(u, v, self.get_ee_default_quat())


class RobotControllerExecutionFailed(CrowControllerExecutionError):
    pass


class RobotControllerExecutionContext(object):
    def __init__(self, controller_name, *args, **kwargs):
        self.controller_name = controller_name
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def monitor(self, rv: bool):
        if not rv:
            raise RobotControllerExecutionFailed(f"Execution of {self.controller_name} failed. Args: {self.args}, kwargs: {self.kwargs}")


class RobotArmJointTrajectory(list):
    pass


class RobotControllerInterfaceBase(object):
    def __init__(self):
        self._default_parameters = dict()

    @property
    def default_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get the default parameters for each primitive."""
        return self._default_parameters

    def set_default_parameters(self, primitive: str, **kwargs):
        """Set the default parameters for a specific primitive."""
        if  primitive not in self._default_parameters:
            self._default_parameters[primitive] = dict()
        self._default_parameters[primitive].update(kwargs)

    def get_update_default_parameters(self, primitive: str, kwargs_: Dict[str, Any]) -> None:
        """Update the default parameters for a specific primitive."""
        if primitive not in self._default_parameters:
            return
        kwargs_.update(self._default_parameters[primitive])


class SingleArmControllerInterface(RobotControllerInterfaceBase):
    def get_qpos(self) -> np.ndarray:
        raise NotImplementedError

    def get_qvel(self) -> np.ndarray:
        raise NotImplementedError

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def move_home(self) -> bool:
        raise NotImplementedError

    def move_qpos(self, qpos: np.ndarray) -> None:
        raise NotImplementedError

    def move_qpos_trajectory(self, trajectory: RobotArmJointTrajectory) -> None:
        raise NotImplementedError

    def open_gripper(self) -> None:
        raise NotImplementedError

    def close_gripper(self) -> None:
        raise NotImplementedError

    def grasp(self, width: float = 0.05, force: float = 40) -> None:
        raise NotImplementedError

    def move_grasp(self, approaching_trajectory: RobotArmJointTrajectory, width: float = 0.05, force: float = 40) -> None:
        self.move_qpos_trajectory(approaching_trajectory)
        self.grasp(width, force)

    def move_place(self, placing_trajectory: RobotArmJointTrajectory) -> None:
        self.move_qpos_trajectory(placing_trajectory)
        self.open_gripper()


def _l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return np.linalg.norm(a - b, ord=1)


class CollisionChecker:
    def __init__(
        self,
        check_collision_fn,
        ignored_collision_bodies: Optional[List[Union[str, int]]] = None,
        collision_tol: Optional[float] = None,
        move_from_contact_collision_args: Optional[MotionPlanningContactTimeCollisionCheckingArguments] = None,
        move_to_contact_collision_args: Optional[MotionPlanningContactTimeCollisionCheckingArguments] = None,
        move_from_qpos: Optional[VecNf] = None,
        move_to_qpos: Optional[VecNf] = None,
    ):
        self.check_collision_fn = check_collision_fn
        self.ignored_collision_bodies = ignored_collision_bodies
        self.collision_tol = collision_tol
        self.move_from_contact_collision_args = move_from_contact_collision_args
        self.move_to_contact_collision_args = move_to_contact_collision_args
        self.move_from_qpos = move_from_qpos
        self.move_to_qpos = move_to_qpos

    def __call__(self, qpos: VecNf) -> bool:
        current_collision_tol = self.collision_tol
        if self.move_from_contact_collision_args is not None:
            if _l1_distance(qpos, self.move_from_qpos) < self.move_from_contact_collision_args.qpos_range:
                current_collision_tol = self.move_from_contact_collision_args.collision_tol
        if self.move_to_contact_collision_args is not None:
            if _l1_distance(qpos, self.move_to_qpos) < self.move_to_contact_collision_args.qpos_range:
                current_collision_tol = self.move_to_contact_collision_args.collision_tol
        return self.check_collision_fn(qpos, ignored_collision_bodies=self.ignored_collision_bodies, max_distance=current_collision_tol)
