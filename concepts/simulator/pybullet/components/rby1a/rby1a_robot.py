#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : rby1a_robot.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/17/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional, Union, Tuple, Dict

import numpy as np
import pybullet as pb

from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.pybullet.world import BulletSaver, BulletWorld, ConstraintInfo
from concepts.simulator.pybullet.components.robot_base import BulletMultiChainRobotRobotBase
from concepts.simulator.urdf_utils.scipy_ik import scipy_inverse_kinematics
from concepts.algorithm.configuration_space import BoxConfigurationSpace
from concepts.math.range import Range
from concepts.math.rotationlib_xyzw import rotate_vector, quat_mul, quat_conjugate
from concepts.math.frame_utils_xyzw import compose_transformation, inverse_transformation, get_transform_a_to_b
from concepts.utils.typing_utils import Vec3f, Vec4f


class RBY1ARobot(BulletMultiChainRobotRobotBase):
    RBY1A_FILE = 'assets://robots/rby1a/model.urdf'
    RBY1A_TRACIK_FILE = 'assets://robots/rby1a/model_tracikpy.urdf'
    IGNORED_COLLISION_PAIRS = [
        ('link_torso_4', 'link_torso_2'),
        ('link_right_arm_3', 'link_right_arm_5'),
        ('link_left_arm_3', 'link_left_arm_5'),
    ]

    def __init__(
        self, client: BulletClient, body_id: Optional[int] = None, body_name: str = 'rby1a',
        pos: Optional[Vec3f] = None, quat: Optional[Vec4f] = None,
        use_magic_gripper: bool = True, enforce_torso_safety: bool = True
    ):
        super().__init__(client, body_name=body_name, use_magic_gripper=use_magic_gripper)

        if body_id is not None:
            self.body_id = body_id
        else:
            self.body_id = client.load_urdf(self.RBY1A_FILE, pos=pos, quat=quat, static=True, body_name=body_name)

        self.enforce_torso_safety = enforce_torso_safety

        self._init_joints()
        self._tracik_solver = None

        self.world.register_managed_interface('RBY1ARobot@' + str(self.body_id), self)
        self.world.register_additional_state_saver(
            'RBY1ARobot@' + str(self.body_id),
            lambda: RBY1ARobotMagicGripperStateSaver(self.client.client_id, self.world, 'RBY1ARobot@' + str(self.body_id))
        )

        self._self_collision_exclude_link_pairs = list()
        for link_a_name, link_b_name in self.IGNORED_COLLISION_PAIRS:
            self.add_ignore_collision_pair_by_name(link_a_name, link_b_name)

    def is_qpos_valid(self, qpos: np.ndarray):
        torso_joints = qpos[2:2 + 6]
        state = torso_joints[1] + torso_joints[2] + torso_joints[3]
        return abs(state) < 0.4

    def get_joint_limits(self, group_name: str = '__all__') -> Tuple[np.ndarray, np.ndarray]:
        """Get the joint limits."""
        body_id = self.get_body_id()
        joint_info = [self.client.w.get_joint_info_by_id(body_id, i) for i in self.get_joint_ids(group_name)]
        lower_limits = np.array([joint.joint_lower_limit for joint in joint_info])
        upper_limits = np.array([joint.joint_upper_limit for joint in joint_info])
        lower_limits[:2] = -10
        upper_limits[:2] = +10
        return lower_limits, upper_limits

    def add_ignore_collision_pair_by_id(self, link_a_id: int, link_b_id: int):
        self._self_collision_exclude_link_pairs.append((link_a_id, link_b_id))
        self._self_collision_exclude_link_pairs.append((link_b_id, link_a_id))

    def add_ignore_collision_pair_by_name(self, link_name_a, link_name_b):
        _, link_a_id = self.client.world.get_link_index(link_name_a)
        _, link_b_id = self.client.world.get_link_index(link_name_b)
        self.add_ignore_collision_pair_by_id(link_a_id, link_b_id)

    def get_body_id(self) -> int:
        return self.body_id

    def get_ee_to_tool(self, hand: str, tool_id: int) -> Tuple[Vec3f, Vec4f]:
        ee_link_id = self.get_ee_link_id(hand)
        robot_pos, robot_quat = self.world.get_link_state_by_id(self.get_body_id(), ee_link_id, fk=True).get_transformation()
        tool_pos, tool_quat = self.world.get_body_state_by_id(tool_id).get_transformation()
        return get_transform_a_to_b(robot_pos, robot_quat, tool_pos, tool_quat)

    def _init_joints(self):
        joints = self.world.get_joint_info_by_body(self.body_id)
        non_stationary_joints = [j for j in joints if j.joint_type != pb.JOINT_FIXED]

        base_joints = [j for j in non_stationary_joints if j.joint_name.endswith(b'wheel')]
        torso_joints = [j for j in non_stationary_joints if j.joint_name.startswith(b'torso')]
        right_joints = [j for j in non_stationary_joints if j.joint_name.startswith(b'right') and b'wheel' not in j.joint_name]
        left_joints = [j for j in non_stationary_joints if j.joint_name.startswith(b'left') and b'wheel' not in j.joint_name]
        head_joints = [j for j in non_stationary_joints if j.joint_name.startswith(b'head')]

        right_gripper_joints = [j for j in non_stationary_joints if j.joint_name.startswith(b'gripper_finger_r')]
        left_gripper_joints = [j for j in non_stationary_joints if j.joint_name.startswith(b'gripper_finger_l')]

        assert len(base_joints) == 2
        assert len(torso_joints) == 6
        assert len(left_joints) == 7
        assert len(right_joints) == 7
        assert len(head_joints) == 2

        def get_ids(joints):
            return [j.joint_index for j in joints]

        self.define_joint_groups('base', get_ids(base_joints), start_index=0)
        self.define_joint_groups(
            'torso', get_ids(torso_joints),
            ee_link_id=self.world.get_link_index_with_body(self.body_id, 'link_torso_5'), start_index=non_stationary_joints.index(torso_joints[0])
        )
        self.define_joint_groups(
            'right', get_ids(right_joints),
            ee_link_id=self.world.get_link_index_with_body(self.body_id, 'ee_right_tool'), start_index=non_stationary_joints.index(right_joints[0])
        )
        self.define_joint_groups(
            'left', get_ids(left_joints),
            ee_link_id=self.world.get_link_index_with_body(self.body_id, 'ee_left_tool'), start_index=non_stationary_joints.index(left_joints[0])
        )
        self.define_joint_groups(
            'head', get_ids(head_joints),
            ee_link_id=self.world.get_link_index_with_body(self.body_id, 'rby1_camera_mount'), start_index=non_stationary_joints.index(head_joints[0])
        )
        self.define_joint_groups(
            'torso_and_right', get_ids(torso_joints + right_joints), ee_link_id=self.world.get_link_index_with_body(self.body_id, 'ee_right_tool'),
            start_index=get_ids(torso_joints + right_joints)
        )
        self.define_joint_groups(
            'torso_and_left', get_ids(torso_joints + left_joints), ee_link_id=self.world.get_link_index_with_body(self.body_id, 'ee_left_tool'),
            start_index=get_ids(torso_joints + right_joints)
        )
        self.define_joint_groups('__all__', get_ids(non_stationary_joints))

        self.define_joint_groups('right_gripper', get_ids(right_gripper_joints), start_index=non_stationary_joints.index(right_gripper_joints[0]))
        self.define_joint_groups('left_gripper', get_ids(left_gripper_joints), start_index=non_stationary_joints.index(left_gripper_joints[0]))

    def ik_pybullet(
        self, pos: np.ndarray, quat: np.ndarray, link_id: int, force: bool = False,
        pos_tol: float = 1e-2, quat_tol: float = 1e-2, verbose: bool = False
    ) -> Optional[Union[np.ndarray, Dict[str, np.ndarray]]]:
        lower_limits, upper_limits = self.get_joint_limits()
        rest_qpos = self.get_qpos()
        joints = self.client.p.calculateInverseKinematics(
            bodyUniqueId=self.get_body_id(),
            endEffectorLinkIndex=link_id,
            targetPosition=pos,
            targetOrientation=quat,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=upper_limits - lower_limits,
            restPoses=rest_qpos,
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        joints = np.asarray(joints)

        if verbose:
            print('IK (solution, lower, upper):\n', np.stack([joints[:7], lower_limits, upper_limits], axis=0), sep='')
        return joints

        # TODO: implement the following checks

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

    def is_self_collision(self, qpos: np.ndarray) -> bool:
        """
        Check if the robot is in self collision
        """
        current_state = self.get_qpos()
        self.set_qpos(qpos)
        contacts = self.world.get_contact(self.body_id, update=True)
        try:
            for contact in contacts:
                if contact.body_b == self.body_id:
                    if (contact.link_a, contact.link_b) in self._self_collision_exclude_link_pairs:
                        continue
                    # print('Collision detected:')
                    # print('> ', contact.body_a_name, contact.link_a_name, contact.body_b_name, contact.link_b_name)
                    return True
            return False
        finally:
            self.set_qpos(current_state)

    def is_valid_ik_solution(self, ik_solution: Dict[str, float], last_state: np.ndarray):
        qpos = self.set_index_full_joint_state_by_name(last_state, ik_solution)
        if not self.is_qpos_valid(qpos):
            return False
        if self.is_self_collision(qpos):
            return False
        return True

    def get_tracik_solver(self):
        from concepts.simulator.tracik_utils.tracik_wrapper import URDFTracIKWrapper

        if self._tracik_solver is None:
            self._tracik_solver = URDFTracIKWrapper(self.client.canonicalize_asset_path(self.RBY1A_TRACIK_FILE), base_link_name='world')
        return self._tracik_solver

    def ik_tracik(self, pos: np.ndarray, quat: np.ndarray, ee_link_id: int, last_qpos: Optional[np.ndarray] = None, max_distance: float = float('inf'), max_attempts: int = 50, verbose: bool = False) -> Optional[np.ndarray]:
        body_pos, body_quat = self.get_body_pose()
        world2base = (body_pos, body_quat)
        world2ee = (pos, quat)
        base2ee = compose_transformation(*inverse_transformation(*world2base), *world2ee)
        link_state = self.world.get_link_state_by_id(self.body_id, ee_link_id, fk=False)
        local_transformation = link_state.local_pos, link_state.local_quat
        base2ee_joint = compose_transformation(*base2ee, *inverse_transformation(*local_transformation))
        target_pos, target_quat = base2ee_joint

        ee_link_name = self.world.get_link_name(self.body_id, ee_link_id).split('/')[-1]
        tracik_solver = self.get_tracik_solver()

        ik_solutions = tracik_solver.gen_ik(ee_link_name, target_pos, target_quat, return_all=True)
        if len(ik_solutions) == 0:
            return None

        if last_qpos is None:
            last_qpos = self.get_qpos()
        valid_solutions = list(filter(lambda sol: self.is_valid_ik_solution(sol, last_qpos), ik_solutions))
        if len(valid_solutions) == 0:
            return None

        solution = valid_solutions[0]
        rv = self.set_index_full_joint_state_by_name(last_qpos, solution)

        if verbose:
            for k, v in solution.items():
                print(k, v)
            if True:
                print('world2ee', world2ee),
                print('world2base', world2base),
                print('base2ee', base2ee),
                print('world2ee_prime', self.fk(rv, ee_link_id))

        return rv

    def ik_scipy(self, pos: np.ndarray, quat: np.ndarray, link_id: int, last_qpos: Optional[np.ndarray] = None, max_distance: float = float('inf'), max_attempts: int = 20, verbose: bool = False) -> Optional[np.ndarray]:
        def fk_func(qpos):
            return self.fk(qpos, link_id)

        qpos_backup = self.get_qpos()
        if last_qpos is None:
            last_qpos = qpos_backup

        lower_bound, upper_bound = self.get_joint_limits()

        def sample_func():
            qpos = np.random.uniform(lower_bound, upper_bound)
            return qpos

        solution = scipy_inverse_kinematics(
            fk_func, pos, quat, lower_bound, upper_bound,
            sample_func=sample_func,
            nr_trials=max_attempts,
            verbose=verbose
        )

        self.set_qpos(qpos_backup)

        return solution

    def internal_set_gripper_state(self, hand: str, activate: bool, body_index: Optional[int] = None) -> None:
        assert hand in ['left', 'right']
        ee_link_id = self.get_ee_link_id(hand)

        if not activate:  # Turn gripper off.
            if self.use_magic_gripper:
                if ee_link_id in self.gripper_constraints:
                    self.detach_object(ee_link_id)
        else:  # Turn gripper on.
            if self.use_magic_gripper:
                if body_index is not None:
                    self.create_gripper_constraint(ee_link_id, body_index)

        self.gripper_states[hand] = activate

    def internal_set_gripper_position(self, hand: str, activate: bool):
        assert hand in ['left', 'right']
        gripper_group = f'{hand}_gripper'
        if activate:
            self.world.set_batched_qpos_by_id(self.body_id, self.get_joint_ids(gripper_group), [0.00, 0.00])
        else:
            self.world.set_batched_qpos_by_id(self.body_id, self.get_joint_ids(gripper_group), [-0.05, 0.05])

    def open_gripper_free(self, hand: str, timeout: float = 0.0, force: bool = False) -> bool:
        self.internal_set_gripper_state(hand, False)
        self.internal_set_gripper_position(hand, False)

    def close_gripper_free(self, hand: str, timeout: float = 0.0, force: bool = False) -> bool:
        self.internal_set_gripper_state(hand, True)
        self.internal_set_gripper_position(hand, True)


class RBY1ARobotMagicGripperStateSaver(BulletSaver):
    def __init__(self, client_id: int, world: BulletWorld, managed_interface: str):
        super().__init__(client_id, world)
        self.managed_interface = managed_interface
        self.gripper_states = None
        self.gripper_constraint_info = None

    @property
    def robot(self) -> RBY1ARobot:
        return self.world.managed_interfaces[self.managed_interface]

    def save(self):
        self.gripper_states = self.robot.gripper_states.copy()
        self.gripper_constraints = {ee_id: self.world.get_constraint(c).child_body for ee_id, c in self.robot.gripper_constraints.items()}

    def restore(self):
        for _, cid in self.robot.gripper_constraints.items():
            pb.removeConstraint(cid, physicsClientId=self.client_id)
        self.robot.gripper_constraints = dict()
        self.robot.gripper_states.update(self.gripper_states)
        for ee_id, body_id in self.gripper_constraints.items():
            self.robot.create_gripper_constraint(ee_id, body_id)

