#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pybullet_single_robot_rwi_pp_simplified.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/27/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import itertools
import os.path as osp
from typing import Union, Tuple, Dict

import jacinle
import torch
import numpy as np

from concepts.math.frame_utils_xyzw import solve_ee_from_tool
from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.pybullet.world import BulletWorld
from concepts.simulator.pybullet.components.robot_base import BulletArmRobotBase
from concepts.simulator.pybullet.manipulation_utils.path_generation_utils import gen_qpos_path_from_ee_path
from concepts.simulator.pybullet.qddl_interface import QDDLSceneMetainfo

import concepts.dm.crow as crow
from concepts.dm.crowhat.impl.pybullet.pybullet_sim_interfaces import PyBulletRemotePerceptionInterface, PyBulletSimulationControllerInterface
from concepts.dm.crowhat.world.manipulator_interface import RobotArmJointTrajectory
from concepts.dm.crowhat.impl.pybullet.pybullet_planning_world_interface import PyBulletPlanningWorldInterface
from concepts.dm.crowhat.impl.pybullet.pybullet_manipulator_interface import PyBulletSingleArmMotionPlanningInterface
from concepts.dm.crowhat.manipulation_utils.pick_place_sampler import GraspParameter, gen_grasp_parameter, calc_grasp_approach_ee_pose_trajectory
from concepts.dm.crowhat.manipulation_utils.pick_place_sampler import PlacementParameter, gen_placement_parameter

logger = jacinle.get_logger(__name__)


class PyBulletSingleArmRWIPPSimplifiedDomain(object):
    domain_filename = osp.join(osp.dirname(__file__), 'RWI-pp-simplified.cdl')

    def __init__(self, robot: BulletArmRobotBase, scene_metainfo: QDDLSceneMetainfo, verbose: bool = False, check_z_in_on_test: bool = False, use_specialized_block_placement_sampler: bool = True):
        """Initialize the RWI-PP-Simplified domain.

        Args:
            robot: the robot in the PyBullet environment.
            scene_metainfo: the metainfo of the scene.
            verbose: whether to print verbose information.
            check_z_in_on_test: whether to check object1.z > object2.z when testing "on(object1, object2)".
            use_specialized_block_placement_sampler: whether to use a specialized block placement sampler. When the object to be placed onto is a block,
                we will use a specialized sampler that always return the "center" of the block as the placement position.
        """
        self.robot = robot
        self.world = self.robot.world
        self.client = self.robot.client
        self.metainfo = scene_metainfo
        self.verbose = verbose

        self.world_interface = PyBulletPlanningWorldInterface(self.client)
        self.robot_interface = PyBulletSingleArmMotionPlanningInterface(self.robot, self.world_interface)

        self.domain = crow.load_domain_file(type(self).domain_filename)
        self.executor = crow.CrowExecutor(self.domain)

        self.check_z_in_on_test = check_z_in_on_test
        self.use_specialized_block_placement_sampler = use_specialized_block_placement_sampler

        self.bind_executor_functions()

    world: BulletWorld
    """The PyBullet world."""

    client: BulletClient
    """The PyBullet client."""

    robot: BulletArmRobotBase
    """The Panda robot in the PyBullet environment."""

    metainfo: QDDLSceneMetainfo
    """The metainfo of the scene."""

    verbose: bool
    """Whether to print verbose information."""

    domain: crow.CrowDomain
    """The domain of the RWI-PP-Simplified domain."""

    executor: crow.CrowExecutor
    """The executor of the RWI-PP-Simplified domain."""

    _initial_object_poses: Dict[str, Tuple[np.ndarray, np.ndarray]]

    def bind_perception_interface(self, perception_interface: PyBulletRemotePerceptionInterface):
        perception_interface.register_bullet_client(self.client)
        perception_interface.register_state_getter(self._get_state)

    def bind_simulation_interface(self, simulation_interface: PyBulletSimulationControllerInterface):
        simulation_interface.register_state_getter(self._get_state)

    def _get_state(self, interface: Union[PyBulletRemotePerceptionInterface, PyBulletSimulationControllerInterface]) -> crow.CrowState:
        objects = dict()
        robot_name = None
        for name, info in self.metainfo.objects.items():
            if info.id == self.robot.get_body_id():
                robot_name = name
                object_type = 'Hand'
            else:
                object_type = 'Object'
            objects[name] = object_type

        state = crow.CrowState.make_empty_state(self.domain, objects)

        for name, info in self.metainfo.objects.items():
            index = info.id
            if index == self.robot.get_body_id():
                state.fast_set_value('qpos_of', [name], torch.tensor(self.robot.get_qpos(), dtype=torch.float32))
            else:
                state.fast_set_value('pose_of', [name], torch.tensor(self.world.get_body_state_by_id(index).get_7dpose(), dtype=torch.float32))

        if 'support' in self.domain.features:
            for name, info in self.metainfo.objects.items():
                if info.id != self.robot.get_body_id():
                    for name2 in self.world.get_supporting_objects_by_id(info.id):
                        if self.metainfo.objects[name2].id != self.robot.get_body_id():
                            state.fast_set_value('support', [name, name2], True)

        for name, info in self.metainfo.objects.items():
            if info.moveable:
                state.fast_set_value('moveable', [name], True)

        if hasattr(self.robot, 'gripper_constraint'):
            if self.robot.gripper_constraint is None:
                state.fast_set_value('hand_available', [robot_name], True)
            else:
                constraint = self.world.get_constraint(self.robot.gripper_constraint)
                name = self.world.body_names[constraint.child_body]
                state.fast_set_value('hand_available', [robot_name], False)
                state.fast_set_value('holding', [robot_name, name], True)
        else:
            state.fast_set_value('hand_available', [robot_name], True)

        return state

    def bind_executor_functions(self):
        self._init_object_poses_table()

        @crow.config_function_implementation(support_batch=False)
        def on_with_pose(x, y, x_pose, y_pose):
            if self.check_z_in_on_test:
                rv = (x_pose[0] - y_pose[0]) ** 2 + (x_pose[1] - y_pose[1]) ** 2 < 0.05 and (x_pose[2] - y_pose[2]) > 0.0
            else:
                rv = (x_pose[0] - y_pose[0]) ** 2 + (x_pose[1] - y_pose[1]) ** 2 < 0.05
            return rv

        @crow.config_function_implementation(is_iterator=True)
        def sample_grasp_inner(robot_name, target_name):
            target_id = self.metainfo.get_object_identifier(target_name)
            sampler = gen_grasp_parameter(self.world_interface, self.robot_interface, target_id, 0.08, max_test_points_before_first=200, verbose=False)
            if 'box' in self.world.get_body_name(target_id):
                sampler = itertools.islice(sampler, 5)
            for grasp_param in sampler:
                yield grasp_param

        @crow.config_function_implementation(is_iterator=True)
        def sample_placement_inner(robot_id, target_name, support_name):
            target_id = self.metainfo.get_object_identifier(target_name)
            support_id = self.metainfo.get_object_identifier(support_name)
            if support_id == self.world.get_body_index('table'):
                # NB(Jiayuan Mao @ 2023/04/13): try two different placements: the original placement, and a random placement.
                pos, quat = self._initial_object_poses[target_name]
                x = np.random.rand() * 0.2 + 0.1
                y = np.random.rand() * 0.8 - 0.4
                yield torch.tensor([x, y, 0.1, quat[0], quat[1], quat[2], quat[3]], dtype=torch.float32)
                yield torch.tensor([pos[0], pos[1], 0.1, quat[0], quat[1], quat[2], quat[3]], dtype=torch.float32)
            else:
                if self.use_specialized_block_placement_sampler and self.world.get_collision_shape_data_by_id(support_id)[0].shape_type == 3:
                    target_pos, target_quat = self.world.get_body_state_by_id(target_id).get_transformation()
                    support_pos, support_quat = self.world.get_body_state_by_id(support_id).get_transformation()
                    yield torch.tensor([support_pos[0], support_pos[1], support_pos[2] + 0.05, target_quat[0], target_quat[1], target_quat[2], target_quat[3]], dtype=torch.float32)
                else:
                    for param in gen_placement_parameter(self.world_interface, target_id, support_id, retain_target_orientation=True):
                        yield torch.tensor([param.target_pos[0], param.target_pos[1], param.target_pos[2], param.target_quat[0], param.target_quat[1], param.target_quat[2], param.target_quat[3]], dtype=torch.float32)

        @crow.config_function_implementation(is_iterator=True)
        def sample_grasp_trajectory_inner(robot_name, target_name, param: GraspParameter):
            target_id = self.metainfo.get_object_identifier(target_name)
            trajectory = calc_grasp_approach_ee_pose_trajectory(param, 0.05)
            succ, qpos_trajectory = gen_qpos_path_from_ee_path(self.robot, trajectory)
            if succ:
                # If the trajectory is valid, we then do cfree planning to generate a trajectory from the current qpos to the first qpos.
                succ, qpos_trajectory_first = self.robot.rrt_collision_free(qpos_trajectory[0], smooth_fine_path=True)
                if succ:
                    yield RobotArmJointTrajectory(list(qpos_trajectory_first) + qpos_trajectory[1:])
            else:
                if self.verbose:
                    logger.warning('Failed to generate a valid grasp trajectory.')

        @crow.config_function_implementation(is_iterator=True)
        def sample_placement_trajectory_inner(robot_name, target_name, support_name, tensor_param: torch.Tensor):
            target_id = self.metainfo.get_object_identifier(target_name)
            support_id = self.metainfo.get_object_identifier(support_name)
            param = PlacementParameter(
                object_id=target_id, support_id=support_id,
                target_pos=np.array((tensor_param[0], tensor_param[1], tensor_param[2]), dtype=np.float32),
                target_quat=np.array((tensor_param[3], tensor_param[4], tensor_param[5], tensor_param[6]), dtype=np.float32),
                support_normal=np.array([0, 0, 1], dtype=np.float32),
            )

            ee_to_target = self.robot.get_ee_to_tool(target_id)

            trajectory = list()
            for placement_height in [0.09, 0.08, 0.07, 0.06, 0.05, 0.03, 0.01]:
                robot_start_pos, robot_start_quat = solve_ee_from_tool(param.target_pos + param.support_normal * placement_height, param.target_quat, ee_to_target)
                trajectory.append((robot_start_pos, robot_start_quat))

            succ, qpos_trajectory = gen_qpos_path_from_ee_path(self.robot, trajectory, max_pairwise_distance=0.5)

            if not succ:
                import ipdb; ipdb.set_trace()

            if succ:
                last_qpos = self.robot.get_qpos()
                cfree_qpos_trajectory = list()

                for qpos in qpos_trajectory:
                    succ, cfree_qpos = self.robot.rrt_collision_free(last_qpos, qpos, smooth_fine_path=True)
                    if succ:
                        cfree_qpos_trajectory.extend(cfree_qpos)
                        last_qpos = qpos
                    else:
                        succ = False
                        break

                if succ:
                    yield RobotArmJointTrajectory(cfree_qpos_trajectory)
            else:
                if self.verbose:
                    logger.warning('Failed to generate a valid placement trajectory.')

        @crow.config_function_implementation(support_batch=False)
        def _is_valid_grasp_param(robot_id, target_id, param: GraspParameter):
            return True

        @crow.config_function_implementation(support_batch=False)
        def _is_valid_place_param(robot_id, target_id, support_id, param: torch.Tensor):
            return True

        @crow.config_function_implementation(support_batch=False)
        def _is_valid_grasp_traj(robot_id, target_id, param: GraspParameter, traj: RobotArmJointTrajectory):
            return True

        @crow.config_function_implementation(support_batch=False)
        def _is_valid_place_traj(robot_id, target_id, support_id, param: torch.Tensor, traj: RobotArmJointTrajectory):
            return True

        self.executor.register_function('on_with_pose', on_with_pose)
        self.executor.register_function('gen_valid_grasp', sample_grasp_inner)
        self.executor.register_function('gen_valid_placement', sample_placement_inner)
        self.executor.register_function('gen_valid_grasp_traj', sample_grasp_trajectory_inner)
        self.executor.register_function('gen_valid_placement_traj', sample_placement_trajectory_inner)
        self.executor.register_function('valid_grasp', _is_valid_grasp_param)
        self.executor.register_function('valid_grasp_traj', _is_valid_grasp_traj)
        self.executor.register_function('valid_placement', _is_valid_place_param)
        self.executor.register_function('valid_placement_traj', _is_valid_place_traj)

    def _init_object_poses_table(self):
        self._initial_object_poses = dict()
        for body_id, body_name in self.world.body_names:
            self._initial_object_poses[body_id] = self.world.get_body_state_by_id(body_id).get_transformation()

