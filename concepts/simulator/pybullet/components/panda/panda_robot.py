#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : panda_robot.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/23/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import itertools
import numpy as np
import pybullet as p
from typing import Any, Optional, Union, Iterable, Sequence, Tuple, Dict

from jacinle.logging import get_logger
from jacinle.utils.defaults import ARGDEF, default_args

from concepts.math.interpolation_utils import gen_linear_spline, get_next_target_linear_spline
from concepts.math.rotationlib_xyzw import quat_mul, quat_conjugate, rotate_vector, quat_diff_in_axis_angle
from concepts.math.frame_utils_xyzw import solve_ee_from_tool, get_transform_a_to_b, compose_transformation
from concepts.utils.typing_utils import VecNf

from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.pybullet.world import ConstraintInfo, BulletSaver, BulletWorld
from concepts.simulator.pybullet.components.robot_base import BulletArmRobotBase, GripperObjectIndices, BulletRobotActionPrimitive
from concepts.simulator.pybullet.control_utils import get_os_imp_control_command, get_default_os_imp_control_parameters

logger = get_logger(__file__)

np.set_printoptions(precision=5, suppress=True)

__all__ = [
    'PandaRobot', 'PandaRobotMagicGripperStateSaver',
    'PandaReachTwoStage', 'PandaPlanarPush', 'PandaPushTwoStage', 'PandaPickAndPlace'
]


class PandaRobot(BulletArmRobotBase):
    # PANDA_FILE = 'assets://franka_panda/panda.urdf'
    # PANDA_JOINT_HOMES = np.array([0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32])
    PANDA_FILE = 'assets://robots/franka_description/robots/panda_arm_hand_with_inertial.urdf'
    PANDA_SOFT_FINGER_FILE = 'assets://robots/franka_description_softfinger/robots/franka_panda_softfinger.urdf'
    PANDA_JOINT_HOMES = np.array([-0.45105, -0.38886, 0.45533, -2.19163, 0.13169, 1.81720, 0.51563])

    PANDA_GRIPPER_HOME = 0.04
    PANDA_GRIPPER_OPEN = 0.04
    PANDA_GRIPPER_CLOSE = 0.00

    @classmethod
    def get_available_versions(cls) -> Tuple[str, ...]:
        return ('default', 'soft_finger')

    @classmethod
    def get_metainfo_from_version(cls, version: str) -> Dict[str, Any]:
        if version == 'default':
            return {
                'description_file': cls.PANDA_FILE,
                'gripper_depth': 0.1,
            }
        elif version == 'soft_finger':
            return {
                'description_file': cls.PANDA_SOFT_FINGER_FILE,
                'gripper_depth': 0.15,
            }
        else:
            raise ValueError(f'Unknown version: {version}')

    def __init__(
        self,
        client: BulletClient,
        body_id: Optional[int] = None,
        version: str = 'default',
        body_name: str = 'panda',
        gripper_objects: Optional[GripperObjectIndices] = None,
        use_magic_gripper: bool = True,
        pos: Optional[np.ndarray] = None,
        quat: Optional[np.ndarray] = None,
    ):
        super().__init__(client, body_name, gripper_objects)

        self._version = version
        metainfo = self.get_metainfo_from_version(version)
        self._description_file = metainfo['description_file']
        self._finger_depth = metainfo['gripper_depth']

        if body_id is not None:
            self.panda = body_id
        else:
            self.panda = client.load_urdf(self._description_file, pos=pos, quat=quat, static=True, body_name=body_name, group=None)

        all_joints = client.w.get_joint_info_by_body(self.panda)
        self.panda_joints = [joint.joint_index for joint in all_joints if joint.joint_type == p.JOINT_REVOLUTE]
        self.gripper_joints = [joint.joint_index for joint in all_joints if joint.joint_type == p.JOINT_PRISMATIC]
        self.full_joints = self.panda_joints + self.gripper_joints
        assert set(self.gripper_joints) == {9, 10}
        self.panda_ee_tip = 11
        self.ikfast_wrapper = None

        # for joint_index in self.panda_joints:
        #     self.client.p.changeDynamics(self.panda, joint_index, linearDamping=0.0, angularDamping=0.0)

        # Create a constraint to keep the fingers centered
        # https://github.com/bulletphysics/bullet3/blob/daadfacfff365852ffc96f373c834216a25b11e5/examples/pybullet/gym/pybullet_robots/panda/panda_sim_grasp.py#L46-L54

        c = self.client.p.createConstraint(
            self.panda, 9, self.panda, 10,
            jointType=p.JOINT_GEAR, jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0]
        )
        self.client.p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        joint_info = [self.client.w.get_joint_info_by_id(self.panda, i) for i in self.panda_joints]
        self.panda_joints_lower = np.array([joint.joint_lower_limit for joint in joint_info])
        self.panda_joints_upper = np.array([joint.joint_upper_limit for joint in joint_info])
        self.panda_leftfinger = self.w.link_names[f'{self.body_name}/panda_leftfinger'].link_id
        self.panda_rightfinger = self.w.link_names[f'{self.body_name}/panda_rightfinger'].link_id

        # State variable: gripper_activated. True if the gripper is activated.
        # For all control actions, make sure to call self._set_gripper_control before steping the simulation.
        self.gripper_activated = False

        self.use_magic_gripper = use_magic_gripper
        self.gripper_constraint = None

        if body_id is None:
            self.reset_home_qpos()
        self.home_qpos = np.array(self.PANDA_JOINT_HOMES, dtype=np.float32)
        self.ee_home_pos, self.ee_home_quat = self.get_ee_pose()
        self.ee_default_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        self.reach_two_stage = PandaReachTwoStage(self)
        self.planar_push = PandaPlanarPush(self)
        self.push_two_stage = PandaPushTwoStage(self)
        self.pick_and_place = PandaPickAndPlace(self)

        self._cspace = None
        self._cfree_default_pspace = None

        self.register_action('move_pose', self.move_pose)
        self.register_action('move_home', self.move_home)
        self.register_action('move_home_cfree', self.move_home_cfree)

        self.register_action('grasp', self.grasp)
        self.register_action('open_gripper_free', self.open_gripper_free)
        self.register_action('close_gripper_free', self.close_gripper_free)

        self.register_action('reach_two_stage', self.reach_two_stage)
        self.register_action('planar_push', self.planar_push)
        self.register_action('push_two_stage', self.push_two_stage)
        self.register_action('pick_and_place', self.pick_and_place)
        self.register_action('reach_two_stage', self.reach_two_stage, interface='franka')
        self.register_action('planar_push', self.planar_push, interface='franka')
        self.register_action('pick_and_place', self.pick_and_place, interface='franka')

        self.world.register_managed_interface('PandaRobot@' + str(self.panda), self)
        self.world.register_additional_state_saver(
            'PandaRobot@' + str(self.panda),
            lambda: PandaRobotMagicGripperStateSaver(self.client.client_id, self.world, 'PandaRobot@' + str(self.panda))
        )

    def set_finger_depth(self, depth: float):
        self._finger_depth = depth

    def reset_home_qpos(self):
        self.client.w.set_batched_qpos_by_id(self.panda, self.panda_joints, type(self).PANDA_JOINT_HOMES)
        self.client.w.set_batched_qpos_by_id(self.panda, self.gripper_joints, [type(self).PANDA_GRIPPER_HOME] * 2)
        self.gripper_activated = None

    def get_body_id(self) -> int:
        return self.panda

    def get_joint_ids(self) -> Sequence[int]:
        return self.panda_joints

    def get_home_qpos(self) -> np.ndarray:
        return self.home_qpos

    def get_full_joint_ids(self) -> Sequence[int]:
        return self.full_joints

    def get_full_home_qpos(self) -> np.ndarray:
        return np.concatenate([self.get_home_qpos(), [type(self).PANDA_GRIPPER_HOME] * 2])

    def get_ee_link_id(self) -> int:
        return self.panda_ee_tip

    def get_ee_home_pos(self) -> np.ndarray:
        return self.ee_home_pos

    def get_ee_home_quat(self) -> np.ndarray:
        return self.ee_home_quat

    def get_ee_default_quat(self) -> np.ndarray:
        return self.ee_default_quat

    def get_gripper_body_id(self) -> int:
        return self.panda

    def get_gripper_joint_ids(self) -> Sequence[int]:
        return self.gripper_joints

    def get_gripper_home_qpos(self) -> np.ndarray:
        return np.array([type(self).PANDA_GRIPPER_HOME] * 2)

    def get_gripper_state(self) -> Optional[bool]:
        return self.gripper_activated

    def set_home_qpos(self, qpos: np.ndarray, ee_default_quat: Optional[np.ndarray] = None) -> None:
        self.home_qpos = qpos
        self.ee_home_pos, self.ee_home_quat = self.fk(qpos)
        if ee_default_quat is not None:
            self.ee_default_quat = ee_default_quat

    def attach_object(self, object_id: int, ee_to_object: Tuple[np.ndarray, np.ndarray], use_grasp_function: bool = True) -> None:
        super().attach_object(object_id, ee_to_object)
        # Next, we call the grasp() function to close the gripper. When the gripper hits the object, the object will be attached to the gripper.
        # In this step, it will automatically clear up the previous constraint we created and attach the object to the gripper with a new constraint.
        if use_grasp_function:
            self.grasp()
        else:
            self.gripper_activated = True

    def get_ee_pose_from_attached_object_pose(self, pos: np.ndarray, quat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get the end-effector pose given the desired pose of the attached object."""
        attached_object = self.get_attached_object()
        if attached_object is None:
            raise ValueError('No object is attached to the gripper.')

        world_to_ee = self.get_ee_pose(fk=True)
        world_to_obj = self.world.get_body_state_by_id(attached_object).get_transformation()
        ee_to_tool = get_transform_a_to_b(world_to_ee[0], world_to_ee[1], world_to_obj[0], world_to_obj[1])

        return solve_ee_from_tool(pos, quat, ee_to_tool)

    def _init_ikfast(self):
        if self.ikfast_wrapper is None:
            from concepts.simulator.pybullet.pybullet_ikfast_utils import IKFastPyBulletWrapper
            import concepts.simulator.ikfast.franka_panda.ikfast_panda_arm as ikfast_module
            self.ikfast_wrapper = IKFastPyBulletWrapper(
                self.world, ikfast_module,
                body_id=self.panda,
                joint_ids=self.panda_joints,
                free_joint_ids=[self.panda_joints[-1]],
                use_xyzw=True
            )

    def ikfast(self, pos: np.ndarray, quat: np.ndarray, last_qpos: Optional[np.ndarray] = None, max_attempts: int = 1000, max_distance: float = float('inf'), error_on_fail: bool = True, verbose: bool = False) -> Optional[np.ndarray]:
        self._init_ikfast()

        if verbose:
            print('Solving IK for pos:', pos, 'quat:', quat)

        # NB(Jiayuan Mao @ 2023/08/09): relative transformation between the base and the end-effector.
        pos_delta = [0, 0, self._finger_depth]
        # quat_delta = (0, 0, 1, 0)
        # quat_delta = (0, 0, 1, 0)
        quat_delta = (0.0, 0.0, 0.9238795325108381, 0.38268343236617297)
        inner_quat = quat_mul(quat, quat_conjugate(quat_delta))
        inner_pos = np.array(pos) - rotate_vector(pos_delta, quat)

        body_state = self.world.get_body_state_by_id(self.panda)
        try:
            ik_solution = list(itertools.islice(self.ikfast_wrapper.gen_ik(
                inner_pos, inner_quat, last_qpos=last_qpos, max_attempts=max_attempts, max_distance=max_distance,
                body_pos=body_state.position, body_quat=body_state.orientation, verbose=False
            ), 1))[0]
        except IndexError:
            if error_on_fail:
                raise
            return None

        if verbose:
            print('IK (solution, lower, upper):\n', np.stack([ik_solution, self.panda_joints_lower, self.panda_joints_upper], axis=0), sep='')
            print('FK:', self.fk(ik_solution))

        return ik_solution

    def ikfast_fk(self, q: np.ndarray, link_id: int = 8):
        assert link_id in (8, 11), 'Only support link_id=8 or 11.'

        self._init_ikfast()
        pos, quat = self.ikfast_wrapper.fk(q)
        pos, quat = np.array(pos), np.array(quat)

        if link_id == 8:
            dq = [0.00000000, 0.00000000, 0.38268343, -0.92387953]
            quat = quat_mul(quat, dq)
            return pos, quat
        elif link_id == 11:
            dp = [0, 0, 0.1]
            dq = [0.0, 0.0, 0.9238795325108381, 0.38268343236617297]
            return compose_transformation(pos, quat, dp, dq)
        else:
            assert False, 'Invalid link_id.'

    def is_colliding(self, q: Optional[np.ndarray] = None, return_contacts: bool = False, ignored_collision_bodies: Optional[Sequence[int]] = None):
        """Check if the robot is colliding with other objects. When the joint configuration (q) is provided, we will set the robot to that configuration before checking the collision.
        Note that this function will not restore the robot to the original configuration after the check. If you want to restore the robot to the original configuration,
        you should use :meth:`is_colliding_with_saved_state` instead.
        """
        if q is not None:
            self.set_qpos_with_holding(q)
        contacts = self.world.get_contact(self.panda, update=True)

        ignore_bodies = [self.panda]
        if self.gripper_constraint is not None:
            constraint = self.p.getConstraintInfo(self.gripper_constraint)
            other_body_id = constraint[2]
            contacts += self.world.get_contact(other_body_id, update=True)
            ignore_bodies.append(other_body_id)
            # print('other_body_id:', other_body_id, 'pose:', self.world.get_body_state_by_id(other_body_id).get_transformation())
        if ignored_collision_bodies is not None:
            ignore_bodies.extend(ignored_collision_bodies)

        filtered_contacts = list()
        for c in contacts:
            if c.body_b not in ignore_bodies:
                filtered_contacts.append(c)
                # jacinle.log_function.print('Detected collision between', c.body_a_name, 'and', c.body_b_name)
                if not return_contacts:
                    return True

        return False if not return_contacts else filtered_contacts

    @default_args
    def set_arm_joint_position_control(self, target_qpos: np.ndarray, control_mode: int = p.POSITION_CONTROL, gains: float = 0.3, set_gripper_control: bool = True):
        vector_gains = np.ones(len(self.panda_joints)) * gains
        self.client.p.setJointMotorControlArray(
            bodyIndex=self.panda,
            jointIndices=self.panda_joints,
            controlMode=control_mode,
            targetPositions=target_qpos,
            positionGains=vector_gains,
        )

        if set_gripper_control:
            self.set_gripper_control()

    @default_args
    def set_ee_impedance_control(
        self, target_pos: np.ndarray, target_quat: np.ndarray,
        kp_pos: Union[float, VecNf] = 200, kp_ori: Union[float, VecNf] = 1,
        kd_pos: Optional[Union[float, VecNf]] = None, kd_ori: Optional[Union[float, VecNf]] = 0.01, max_torque: float = 100,
        damping_scale: float = 2.0,
        simulate_with_position_pd: bool = False,
        tau_to_qpos_ratio: float = 0.0005,
        set_gripper_control: bool = True,
        verbose: bool = True,
    ):
        config = get_default_os_imp_control_parameters(kp_pos, kp_ori, kd_pos=kd_pos, kd_ori=kd_ori, damping_scale=damping_scale)
        curr_pos, curr_quat = self.get_ee_pose()
        curr_vel, curr_omg = self.get_ee_velocity()
        J = self.get_jacobian()

        tau = get_os_imp_control_command(curr_pos, curr_quat, target_pos, target_quat, curr_vel, curr_omg, J, config)

        # config['P_ori'][:] = 1.0
        # config['D_ori'][:] = 0.01

        # F_p = np.concatenate([config['P_pos'] * delta_pos, config['P_ori'] * delta_ori])
        # F_d = np.concatenate([config['D_pos'] * curr_vel, config['D_ori'] * curr_omg])
        # F = F_p - F_d  # Equivalent to F = Kp * (x_d - x) + Kd * (0 - x_dot)
        # tau = np.dot(J.T, F)

        if verbose:
            delta_pos = (target_pos - curr_pos)
            delta_ori = quat_diff_in_axis_angle(target_quat, curr_quat)

            print('EE pos:', tuple(curr_pos), 'quat:', tuple(curr_quat))
            print('EE vel:', tuple(curr_vel), 'omg: ', tuple(curr_omg))
            print('Delta pos:', tuple(delta_pos), 'norm', np.linalg.norm(delta_pos), 'delta ori:', tuple(delta_ori), 'norm', np.linalg.norm(delta_ori))
            print('Tau norm:', np.linalg.norm(tau, ord=2))
            # input('Press Enter to continue...')

        if simulate_with_position_pd:
            tau = np.clip(tau, -max_torque, max_torque)
            qpos = self.get_qpos()
            qpos += tau * tau_to_qpos_ratio

            self.client.p.setJointMotorControlArray(
                bodyIndex=self.panda,
                jointIndices=self.panda_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=qpos,
                positionGains=np.ones(len(self.panda_joints)) * 1.0,
            )
        else:
            tau += self.get_coriolis_torque()
            tau = np.clip(tau, -max_torque, max_torque)

            self.client.p.setJointMotorControlArray(
                bodyIndex=self.panda,
                jointIndices=self.panda_joints,
                controlMode=p.VELOCITY_CONTROL,
                forces=np.zeros_like(tau),
            )
            self.client.p.setJointMotorControlArray(
                bodyIndex=self.panda,
                jointIndices=self.panda_joints,
                controlMode=p.TORQUE_CONTROL,
                forces=tau,
            )

        if set_gripper_control:
            self.set_gripper_control()

    def set_gripper_control(self, target: Optional[float] = None):
        if target is None:
            target = type(self).PANDA_GRIPPER_CLOSE if self.gripper_activated else type(self).PANDA_GRIPPER_OPEN

        for joint_index in self.gripper_joints:
            self.client.p.setJointMotorControl2(
                self.panda, joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=40
            )

    @default_args
    def move_qpos_set_control(self, target_qpos, speed=0.01, timeout=10.0, gains=ARGDEF, local_smoothing: bool = True):
        """Move the robot to the target joint configuration.

        Args:
            target_qpos: the target joint configuration.
            speed: the maximum speed of the robot.
            timeout: the maximum time allowed for the robot to reach the target joint configuration.
            gains: the gains for the joint position control.
            local_smoothing: if True, the robot will move to the target joint configuration smoothly. This is implemented by setting intermediate joint
                configurations between the current joint configuration and the target joint configuration. If False, the PD controller will directly move
                the robot to the target joint configuration.
        """
        last_qpos = None
        for _ in self.client.timeout(timeout):
            current_qpos = self.client.w.get_batched_qpos_by_id(self.panda, self.panda_joints)
            diff = target_qpos - current_qpos
            norm = np.linalg.norm(diff, ord=2)

            rel_diff, rel_norm = None, 1e9
            if last_qpos is not None:
                rel_diff = last_qpos - current_qpos
                rel_norm = np.linalg.norm(rel_diff, ord=1)
            else:
                last_qpos = current_qpos
                rel_norm = 1e9

            # pos, quat = self.get_ee_pose()
            # print('  current pos', pos, 'quat', quat)
            # print('  current/target joint state\n', np.stack([current_qpos, target_qpos], axis=0), sep='')
            # print('  diff', diff, 'norm', norm)
            # if last_qpos is not None:
            #     print('  rel_diff', rel_diff, 'rel_norm', rel_norm)

            if norm < 0.01:
                return True

            # if rel_norm < 0.001:
            #     return False

            if local_smoothing:
                # If speed is bigger than norm, we clip it to norm.
                step_qpos = current_qpos + diff / norm * min(speed, norm)
            else:
                step_qpos = target_qpos

            self.set_arm_joint_position_control(step_qpos, gains=gains, set_gripper_control=True)
            yield

        if not self.warnings_suppressed:
            logger.warning(f'{self.body_name}: Moving timeout ({timeout}s).')
        return False

    def move_qpos(self, target_qpos, speed=ARGDEF, timeout=ARGDEF, gains=ARGDEF, local_smoothing: bool = True) -> bool:
        try:
            for _ in self.move_qpos_set_control(target_qpos, speed=speed, timeout=timeout, gains=gains, local_smoothing=local_smoothing):
                self.client.step()
            return True
        except StopIteration as e:
            return e.value

    @default_args
    def move_qpos_path_v2_set_control(
        self, qpos_path: Iterable[np.ndarray],
        step_size: float = 1, gains: float = 0.3,
        atol: float = 0.03, timeout: float = 20,
        verbose: bool = False,
        return_world_states: bool = False,
    ):
        qpos_path = dedup_qpos_path(qpos_path)
        spl = gen_linear_spline(qpos_path)

        prev_qpos = None
        prev_qpos_not_moving = 0
        next_id = None

        world_states = []
        for _ in self.client.timeout(timeout):
            current_qpos = self.client.w.get_batched_qpos_by_id(self.panda, self.panda_joints)
            # next_target = get_next_target_cubic_spline(spl, current_qpos, step_size, qpos_trajectory)
            next_id, next_target = get_next_target_linear_spline(
                spl, current_qpos, step_size,
                minimum_x=next_id - step_size + 0.2 if next_id is not None else None
            )

            if verbose:
                print('this step size', np.linalg.norm(next_target - current_qpos, ord=1))
                print('next_id', next_id)
                print('this_target (lower, current, next, upper)\n', np.stack([
                    self.panda_joints_lower,
                    current_qpos,
                    next_target,
                    self.panda_joints_upper,
                ]), sep='')

            last_norm = np.linalg.norm(qpos_path[-1] - current_qpos, ord=1)

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
                        if not self.warnings_suppressed:
                            logger.warning(f'{self.body_name}: No progress for 10 steps.')
                        return False
            prev_qpos = current_qpos

            if last_norm < atol:
                if return_world_states:
                    return world_states
                return True

            self.set_arm_joint_position_control(next_target, gains=gains, set_gripper_control=True)
            yield
            if return_world_states:
                world_states.append(self.world.save_world())

        if not self.warnings_suppressed:
            logger.warning(f'{self.body_name}: Moving timeout ({timeout}s).')
        return False

    def move_qpos_path_v2(
        self, qpos_path: Iterable[np.ndarray],
        step_size: float = ARGDEF, gains: float = ARGDEF,
        atol: float = ARGDEF, timeout: float = ARGDEF,
        verbose: bool = False, return_world_states: bool = False,
    ):
        try:
            for _ in self.move_qpos_path_v2_set_control(
                qpos_path, step_size=step_size, gains=gains, atol=atol, timeout=timeout,
                verbose=verbose, return_world_states=return_world_states
            ):
                self.client.step()
            return True
        except StopIteration as e:
            return e.value

    def move_cartesian_trajectory_set_control(
        self, pose_trajectory: Iterable[Tuple[np.ndarray, np.ndarray]],
        kp_pos: Union[float, VecNf] = ARGDEF, kp_ori: Union[float, VecNf] = ARGDEF,
        kd_pos: Optional[Union[float, VecNf]] = ARGDEF, kd_ori: Optional[Union[float, VecNf]] = ARGDEF,
        max_torque: float = ARGDEF, tau_to_qpos_ratio=ARGDEF
    ):
        for target_pos, target_quat in pose_trajectory:
            self.set_ee_impedance_control(
                target_pos, target_quat, kp_pos, kp_ori, kd_pos, kd_ori,
                max_torque=max_torque, tau_to_qpos_ratio=tau_to_qpos_ratio, set_gripper_control=True
            )
            yield

    def move_cartesian_trajectory(
        self, pose_trajectory: Iterable[Tuple[np.ndarray, np.ndarray]],
        kp_pos: Union[float, VecNf] = ARGDEF, kp_ori: Union[float, VecNf] = ARGDEF,
        kd_pos: Optional[Union[float, VecNf]] = ARGDEF, kd_ori: Optional[Union[float, VecNf]] = ARGDEF,
        max_torque: float = ARGDEF, tau_to_qpos_ratio: float = ARGDEF,
    ):
        try:
            for _ in self.move_cartesian_trajectory_set_control(
                pose_trajectory, kp_pos=kp_pos, kp_ori=kp_ori, kd_pos=kd_pos, kd_ori=kd_ori,
                max_torque=max_torque, tau_to_qpos_ratio=tau_to_qpos_ratio
            ):
                self.client.step()
            return True
        except StopIteration as e:
            return e.value

    def internal_set_gripper_state(self, activate: bool, constraint_info: Optional[ConstraintInfo] = None, body_index: Optional[int] = None) -> None:
        if not activate:  # Turn gripper off.
            if self.use_magic_gripper:
                if self.gripper_constraint is not None:
                    self.detach_object()
        else:  # Turn gripper on.
            if self.use_magic_gripper:
                if constraint_info is not None:
                    if body_index is not None:
                        self.create_gripper_constraint(body_index)
                    else:
                        assert constraint_info is not None
                        self.create_gripper_constraint(constraint_info.child_body)

        self.gripper_activated = activate

    def open_gripper_free(self, timeout: float = ARGDEF, force: bool = False) -> bool:
        return self._change_gripper_state_free(False, timeout=timeout, force=force)

    def close_gripper_free(self, timeout: float = ARGDEF, force: bool = False) -> bool:
        return self._change_gripper_state_free(True, timeout=timeout, force=force)

    def open_gripper_free_set_control(self, timeout: float = ARGDEF, force: bool = False):
        return self._change_gripper_state_free_set_control(False, timeout=timeout, force=force)

    def close_gripper_free_set_control(self, timeout: float = ARGDEF, force: bool = False):
        return self._change_gripper_state_free_set_control(True, timeout=timeout, force=force)

    @default_args
    def _change_gripper_state_free_set_control(self, activate: bool, timeout: float = 2.0, atol: float = 0.002, force: bool = False, verbose: bool = False):
        """A helper function that changes the gripper state assuming no contact with other objects.

        Args:
            activate: True to activate the gripper, False to release it.
            timeout: the timeout for the gripper to reach the target state.
            atol: the tolerance for the gripper to reach the target state.
            force: if True, the gripper will be forced to move to the target state. even if it is already in the target state.
            verbose: if True, print verbose information.
        """

        if self.gripper_activated is not None and activate == self.gripper_activated and not force:
            return True

        if verbose:
            logger.info(f'{self.body_name}: Change gripper state to {activate}, use_magic_gripper={self.use_magic_gripper}, gripper_constraint={self.gripper_constraint}')

        if not activate:
            if self.use_magic_gripper:
                if self.gripper_constraint is not None:
                    self.client.p.removeConstraint(self.gripper_constraint)
                    self.gripper_constraint = None

        self.gripper_activated = activate
        target = type(self).PANDA_GRIPPER_CLOSE if activate else type(self).PANDA_GRIPPER_OPEN
        target_qpos = np.array([target, target])
        arm_qpos = self.get_qpos()

        for _ in self.client.timeout(timeout):
            current_qpos = self.client.w.get_batched_qpos_by_id(self.panda, self.gripper_joints)
            diff = target_qpos - current_qpos
            # print('current_qpos', current_qpos, 'target_qpos', target_qpos, 'diff', diff, 'atol', atol)

            if all(np.abs(diff) < atol):
                return True

            self.set_arm_joint_position_control(arm_qpos, set_gripper_control=True)
            yield

        if not self.warnings_suppressed:
            logger.warning(f'{self.body_name}: Moving gripper timeout ({timeout}s).')
        return True

    @default_args
    def _change_gripper_state_free(self, activate: bool, timeout: float = ARGDEF, atol: float = ARGDEF, force: bool = False, verbose: bool = False) -> bool:
        """A helper function that changes the gripper state assuming no contact with other objects.

        Args:
            activate: True to activate the gripper, False to release it.
            timeout: the timeout for the gripper to reach the target state.
            atol: the tolerance for the gripper to reach the target state.
            force: if True, the gripper will be forced to move to the target state. even if it is already in the target state.
            verbose: if True, print verbose information.

        Returns:
            bool: True if the gripper state changed, False otherwise.
        """
        try:
            for _ in self._change_gripper_state_free_set_control(activate, timeout=timeout, atol=atol, force=force, verbose=verbose):
                self.client.step()
            return True
        except StopIteration as e:
            return e.value

    @default_args
    def grasp_set_control(self, timeout: float = 2, target_object: Optional[int] = None, force_constraint: bool = False, verbose: bool = False):
        target_qpos = np.array([type(self).PANDA_GRIPPER_CLOSE] * 2)
        self.gripper_activated = True

        for _ in self.client.timeout(timeout):
            current_qpos = self.w.get_batched_qpos_by_id(self.panda, self.gripper_joints)
            diff = target_qpos - current_qpos

            if self.detect_gripper_contact(verbose=verbose):
                return True

            if all(np.abs(diff) < 1e-3):
                return False

            for joint_index, target in zip(self.gripper_joints, target_qpos):
                self.p.setJointMotorControl2(
                    self.panda, joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target,
                    force=40
                )
            yield

        if force_constraint:
            if target_object is not None and self.use_magic_gripper and self.gripper_constraint is None:
                self.create_gripper_constraint(target_object)

        if not self.warnings_suppressed:
            logger.warning(f'{self.body_name}: Grasping timeout ({timeout}s).')
        return False

    def grasp(self, timeout: float = ARGDEF, target_object: Optional[int] = None, force_constraint: bool = False, verbose: bool = False) -> bool:
        try:
            for _ in self.grasp_set_control(timeout=timeout, target_object=target_object, force_constraint=force_constraint, verbose=verbose):
                self.client.step()
            return True
        except StopIteration as e:
            return e.value

    def detect_gripper_contact(self, verbose: bool = False):
        points1 = self.client.p.getContactPoints(bodyA=self.panda, linkIndexA=self.panda_leftfinger)
        points2 = self.client.p.getContactPoints(bodyA=self.panda, linkIndexA=self.panda_rightfinger)

        objects1 = {point[2] for point in points1 if point[2] != self.panda}
        objects2 = {point[2] for point in points2 if point[2] != self.panda}
        objects_inter = list(set.intersection(objects1, objects2))

        if len(objects_inter) > 0:
            if verbose:
                for obj_id in objects_inter:
                    logger.info(f'{self.body_name}: Contact detected: {obj_id}, name = {self.w.body_names[obj_id]}.')

            if self.use_magic_gripper:
                obj_id = objects_inter[0]
                self.create_gripper_constraint(obj_id)

            return True
        return False


class PandaRobotMagicGripperStateSaver(BulletSaver):
    def __init__(self, client_id: int, world: BulletWorld, managed_interface: str):
        super().__init__(client_id, world)
        self.managed_interface = managed_interface
        self.gripper_activated = None
        self.gripper_constraint_info = None

    @property
    def robot(self) -> PandaRobot:
        return self.world.managed_interfaces[self.managed_interface]

    def save(self):
        self.gripper_activated = self.robot.gripper_activated
        if self.robot.gripper_constraint is not None:
            contact_info = self.world.get_constraint(self.robot.gripper_constraint)
            self.gripper_constraint_info = contact_info

    def restore(self):
        self.robot.internal_set_gripper_state(self.gripper_activated, self.gripper_constraint_info)


class PandaReachTwoStage(BulletRobotActionPrimitive):
    robot: PandaRobot

    def __call__(self, pos: np.ndarray, quat: np.ndarray, height: float = 0.2, speed: float = 0.01) -> bool:
        logger.info(f'{self.robot.body_name}: Moving to pose (two-stage): {pos}, {quat}.')

        self.robot.move_pose(pos + rotate_vector([0, 0, -height], quat), quat, speed=speed)
        for i in reversed(range(10)):
            self.robot.move_pose(pos + rotate_vector([0, 0, -height / 10 * i], quat), quat, speed=speed)


class PandaPushTwoStage(BulletRobotActionPrimitive):
    robot: PandaRobot

    def __call__(self, start_pos: np.ndarray, normal: np.ndarray, distance: float, timeout: float = 10, prepush_height: float = 0.1, speed: float = 0.01) -> bool:
        self.robot.close_gripper_free()

        pos, quat = self.robot.get_ee_pose()
        self.robot.move_pose(start_pos - prepush_height * normal / np.linalg.norm(normal), quat, speed=speed)

        pos, quat = self.robot.get_ee_pose()
        end_pos = start_pos + normal * distance

        for _ in self.client.timeout(timeout):
            diff = end_pos - pos
            if np.linalg.norm(diff) < 0.01:
                return True
            self.robot.move_pose_smooth(pos + diff / np.linalg.norm(diff) * speed, quat, speed=speed)
            pos, _ = self.robot.get_ee_pose()

        if not self.warnings_suppressed:
            logger.info(f'{self.robot.body_name}: Pushing timeout ({timeout}s).')
        return False


class PandaPlanarPush(BulletRobotActionPrimitive):
    robot: PandaRobot

    def __call__(self, target_pos: np.ndarray, timeout: float = 10, speed: float = 0.01, tol: float = 0.01) -> bool:
        target_pos = np.array(target_pos)
        pos, quat = self.robot.get_ee_pose()
        for _ in self.client.timeout(timeout):
            diff = target_pos - pos

            if np.linalg.norm(diff) < tol:
                return True

            self.robot.move_pose_smooth(pos + diff / np.linalg.norm(diff) * speed, quat, speed=speed)
            pos, _ = self.robot.get_ee_pose()

        if not self.warnings_suppressed:
                logger.warning(f'{self.robot.body_name}: Planar push timeout ({timeout}s).')
        return False


class PandaPickAndPlace(BulletRobotActionPrimitive):
    """A helpful pick-and-place primitive."""

    robot: PandaRobot

    def __call__(
        self,
        pick_pos: np.ndarray, pick_quat: np.ndarray,
        place_pos: np.ndarray, place_quat: np.ndarray,
        reach_kwargs: Optional[Dict[str, Any]] = None, grasp_kwargs: Optional[Dict[str, Any]] = None
    ) -> bool:
        pick_pos = np.array(pick_pos)
        pick_quat = np.array(pick_quat)
        pick_after_pos = pick_pos + np.array([0, 0, 0.2])
        place_pos = np.array(place_pos)
        place_quat = np.array(place_quat)
        place_after_pos = place_pos + np.array([0, 0, 0.2])

        if reach_kwargs is None:
            reach_kwargs = {}
        if grasp_kwargs is None:
            grasp_kwargs = {}

        self.robot.do('open_gripper_free')
        self.robot.do('reach_two_stage', pick_pos, pick_quat, **reach_kwargs)
        self.robot.do('grasp', **grasp_kwargs)
        self.robot.do('move_pose', pick_after_pos, pick_quat)
        self.robot.do('move_pose', place_pos, place_quat)
        self.robot.do('open_gripper_free')
        self.robot.do('move_pose', place_after_pos, place_quat)


def dedup_qpos_path(qpos_trajectory: Iterable[np.ndarray]) -> np.ndarray:
    """Canonicalize a qpos trajectory by removing duplicates."""
    qpos_trajectory = np.asarray(qpos_trajectory)
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
