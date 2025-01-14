#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : franka_pick_place_push.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/04/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
import torch
import open3d as o3d

import concepts.dsl.all as T
import concepts.dm.crow as crow
from concepts.dm.crowhat.manipulation_utils.pick_place_sampler import gen_placement_parameter_on_table
from concepts.math.frame_utils_xyzw import solve_ee_from_tool
from concepts.simulator.pybullet.components.panda.panda_robot import PandaRobot

from cogman import CognitionManager
from franka_base import RobotQPosPath


def register_function_implementations(cogman: CognitionManager, executor: crow.CrowExecutor):
    if not cogman.is_simulation_available():
        raise RuntimeError('PyBullet simulation is not available. The pick-place-push functions requires PyBullet simulation.')

    env = cogman.get_simulation_env()

    from concepts.dm.crowhat.impl.pybullet.pybullet_planning_world_interface import PyBulletPlanningWorldInterface
    from concepts.dm.crowhat.impl.pybullet.pybullet_manipulator_interface import PyBulletSingleArmMotionPlanningInterface

    planning_world = PyBulletPlanningWorldInterface(env.client)
    planning_world.add_ignore_collision_pair_by_name('panda/panda_link7', 'panda/panda_hand')
    # planning_world.add_ignore_collision_pair_by_name('workspace-boundary/plane_z_min', 'panda/panda_link0')

    robot: PandaRobot = env.robots[0]
    robot_interface = PyBulletSingleArmMotionPlanningInterface(robot, planning_world)

    from concepts.dm.crowhat.manipulation_utils.pick_place_sampler import gen_grasp_parameter, gen_placement_parameter
    from concepts.dm.crowhat.manipulation_utils.path_generation_utils import gen_collision_free_qpos_path_from_current_qpos_and_ee_pose
    from concepts.dm.crowhat.manipulation_utils.path_generation_utils import gen_collision_free_qpos_path_from_current_qpos_and_ee_path

    def iomin_2d(object1_min, object1_max, object2_min, object2_max):
        intersection = [
            max(0, min(object1_max[0], object2_max[0]) - max(object1_min[0], object2_min[0])),
            max(0, min(object1_max[1], object2_max[1]) - max(object1_min[1], object2_min[1]))
        ]
        intersection_size = intersection[0] * intersection[1]
        return intersection_size / min(
            (object1_max[0] - object1_min[0]) * (object1_max[1] - object1_min[1]),
            (object2_max[0] - object2_min[0]) * (object2_max[1] - object2_min[1])
        )

    def generate_stacking_relationships():
        object_list = cogman.object_pybullet_ids
        object_aabbs: list[tuple[np.ndarray, np.ndarray]] = list()
        object_centers: list[np.ndarray] = list()
        for index in object_list:
            pcd = planning_world.get_object_point_cloud(index)
            aabb = pcd.min(axis=0), pcd.max(axis=0)
            object_aabbs.append(aabb)
            object_centers.append((aabb[0] + aabb[1]) / 2)

        relations = np.zeros((len(object_list), len(object_list)), dtype=bool)
        for i, aabb1 in enumerate(object_aabbs):
            if cogman.object_identifiers[i] == 'table':
                continue

            for j, aabb2 in enumerate(object_aabbs):
                if i == j:
                    continue

                x = iomin_2d(aabb1[0], aabb1[1], aabb2[0], aabb2[1])
                # print(i, j, aabb1, aabb2, x, object_centers[j][2], object_centers[i][2])
                if x > 0.1 and object_centers[j][2] > object_centers[i][2]:
                    relations[j, i] = True   # j is on top of i

        print('Object stacking relationships:')
        object_names = cogman.object_identifiers
        for i, name in enumerate(object_names):
            for j, name2 in enumerate(object_names):
                print('>', name, 'is on top of', name2) if relations[i, j] else None

        return relations

    def generate_on_relationships():
        object_list = cogman.object_pybullet_ids
        object_aabbs: list[tuple[np.ndarray, np.ndarray]] = list()
        object_centers: list[np.ndarray] = list()
        for index in object_list:
            pcd = planning_world.get_object_point_cloud(index)
            aabb = pcd.min(axis=0), pcd.max(axis=0)
            object_aabbs.append(aabb)
            object_centers.append((aabb[0] + aabb[1]) / 2)

        relations = np.zeros((len(object_list), len(object_list)), dtype=bool)
        for i, aabb1 in enumerate(object_aabbs):
            for j, aabb2 in enumerate(object_aabbs):
                if i == j:
                    continue

                x = iomin_2d(aabb1[0], aabb1[1], aabb2[0], aabb2[1])
                # print(i, j, aabb1, aabb2, x, object_centers[j][2], object_centers[i][2])
                if x > 0.1 and object_centers[j][2] > object_centers[i][2]:
                    if cogman.object_identifiers[i] == 'table':
                        if object_aabbs[j][0][2] < 0.03:
                            relations[j, i] = True
                    else:
                        if object_aabbs[j][0][2] - object_aabbs[i][1][2] < 0.03:
                            relations[j, i] = True   # j is on top of i

        print('Object on relationships:')
        object_names = cogman.object_identifiers
        for i, name in enumerate(object_names):
            for j, name2 in enumerate(object_names):
                print('>', name, 'is on top of', name2) if relations[i, j] else None

        # cogman.get_simulation_env().client.set_rendering(True)
        # cogman.get_simulation_env().client.update_viewer_twice()
        # cogman.get_simulation_env().client.wait_for_user('...')

        return relations


    @crow.config_function_implementation(is_iterator=False)
    def get_all_blocking_objects(x):
        object_name_list = cogman.object_identifiers
        mat = generate_stacking_relationships()
        blocking_objects = np.where(mat[:, object_name_list.index(x)])[0]
        rv = T.StateObjectList(
            T.ListType(cogman.domain.types['Object']),
            [T.StateObjectReference(object_name_list[i], i, cogman.domain.types['Object']) for i in blocking_objects]
        )
        # print('Blocking objects for', x, ':', rv)
        return rv

    @crow.config_function_implementation(is_iterator=False)
    def is_blocking(x, o):  # o is blocking x
        object_name_list = cogman.object_identifiers
        mat = generate_stacking_relationships()
        rv = bool(mat[object_name_list.index(o), object_name_list.index(x)])
        # print(o, 'is blocking', x, ':', rv)
        return rv

    @crow.config_function_implementation(is_iterator=False)
    def on(x, y):
        object_name_list = cogman.object_identifiers
        mat = generate_on_relationships()
        rv = bool(mat[object_name_list.index(x), object_name_list.index(y)])
        return rv

    @crow.config_function_implementation(is_iterator=True)
    def sample_grasp(x):
        name_mapping = cogman.get_pybullet_name_mapping()
        name2id = name_mapping.name2id
        index = name2id[x]
        yield from gen_grasp_parameter(planning_world, robot_interface, index, 0.08, surface_pointing_tol=0.7, verbose=False)

    @crow.config_function_implementation(is_iterator=True)
    def sample_grasp_trajectory(x, grasp_param, qpos):
        pos, quat = grasp_param.robot_ee_pose
        env.client.set_rendering(False)
        qpos_trajectory = gen_collision_free_qpos_path_from_current_qpos_and_ee_pose(
            robot_interface, target_pos=pos, target_quat=quat, return_smooth_path=True, verbose=True,
        )
        env.client.set_rendering(True)
        if qpos_trajectory is not None:
            yield RobotQPosPath(qpos_trajectory)
        return

    @crow.config_function_implementation(is_iterator=True)
    def sample_placement(x):
        yield from []

    @crow.config_function_implementation(is_iterator=True)
    def sample_placement_on(x, y):
        name_mapping = cogman.get_pybullet_name_mapping()
        name2id = name_mapping.name2id
        x_index = name2id[x]
        y_index = name2id[y]
        # TODO(Jiayuan Mao @ 2025/01/02): there is a hack for the placement_tol=0.03, which matches a hack in franka_base.py:open_gripper_ctl
        if y == 'table':
            yield from gen_placement_parameter_on_table(planning_world, x_index, y_index, table_x_range=[0.4, 0.7], table_y_range=[-0.2, 0.2], table_z=0.001, verbose=True, placement_tol=0.03)
        else:
            yield from gen_placement_parameter(planning_world, x_index, y_index, retain_target_orientation=True, placement_tol=0.03)

    @crow.config_function_implementation(is_iterator=True)
    def sample_placement_trajectory(x, place_param, qpos):
        name_mapping = cogman.get_pybullet_name_mapping()
        name2id = name_mapping.name2id
        x_index = name2id[x]
        with planning_world.checkpoint_world():
            # robot.set_qpos(np.array(qpos))
            ee_to_target = robot.get_ee_to_tool(x_index)
            placement_height = 0.00

            pos, quat = solve_ee_from_tool(place_param.target_pos + place_param.support_normal * placement_height, place_param.target_quat, ee_to_target)

            # robot.internal_set_gripper_state(True, body_index=x_index)
            qpos_trajectory = gen_collision_free_qpos_path_from_current_qpos_and_ee_pose(
                robot_interface, target_pos=pos, target_quat=quat, return_smooth_path=True, verbose=True,
            )
            # robot.internal_set_gripper_state(False)

        if qpos_trajectory is not None:
           yield RobotQPosPath(qpos_trajectory)

    from concepts.dm.crowhat.manipulation_utils.plannar_push_sampler import PlanarPushParameter, PlanarIndirectPushParameter
    from concepts.dm.crowhat.manipulation_utils.plannar_push_sampler import gen_planar_push_parameter, gen_planar_indirect_push_parameter
    from concepts.dm.crowhat.manipulation_utils.plannar_push_sampler import calc_push_ee_pose_trajectory

    @crow.config_function_implementation(is_iterator=True)
    def sample_push(target: str, support: str):
        name_mapping = cogman.get_pybullet_name_mapping()
        name2id = name_mapping.name2id
        target_index = name2id[target]
        support_index = name2id[support]
        yield from gen_planar_push_parameter(planning_world, robot_interface, target_index, support_index, verbose=True)

    @crow.config_function_implementation(is_iterator=True)
    def sample_push_trajectory(target: str, support: str, push_param: PlanarPushParameter, qpos: torch.Tensor):
        name_mapping = cogman.get_pybullet_name_mapping()
        name2id = name_mapping.name2id
        target_index = name2id[target]
        push_param: PlanarPushParameter

        ee_pose_trajectory = calc_push_ee_pose_trajectory(push_param)
        qpos_trajectory = gen_collision_free_qpos_path_from_current_qpos_and_ee_path(robot_interface, ee_pose_trajectory, return_smooth_path=True, verbose=True, ignored_collision_bodies=[target_index])
        if qpos_trajectory is not None:
            yield RobotQPosPath(qpos_trajectory)

    @crow.config_function_implementation(is_iterator=True)
    def sample_indirect_push(tool, target, support):
        name_mapping = cogman.get_pybullet_name_mapping()
        name2id = name_mapping.name2id
        tool_index = name2id[tool]
        target_index = name2id[target]
        support_index = name2id[support]
        yield from gen_planar_indirect_push_parameter(planning_world, target_index, tool_index, support_index, verbose=True)

    @crow.config_function_implementation(is_iterator=True)
    def sample_indirect_push_trajectory(tool, target, support, indirect_push_param, qpos):
        name_mapping = cogman.get_pybullet_name_mapping()
        name2id = name_mapping.name2id
        tool_index = name2id[tool]
        indirect_push_param: PlanarIndirectPushParameter
        with planning_world.checkpoint_world():
            ee_to_target = robot.get_ee_to_tool(tool_index)
            tool_pose1 = indirect_push_param.tool_pose
            pos1, quat1 = solve_ee_from_tool(tool_pose1[0], tool_pose1[1], ee_to_target)
            tool_pose2 = indirect_push_param.tool_pose[0] + indirect_push_param.push_dir * indirect_push_param.total_push_distance, indirect_push_param.tool_pose[1]
            pos2, quat2 = solve_ee_from_tool(tool_pose2[0], tool_pose2[1], ee_to_target)

            qpos_trajectory = gen_collision_free_qpos_path_from_current_qpos_and_ee_path(robot_interface, [(pos1, quat1), (pos2, quat2)], return_smooth_path=True, verbose=True)
            if qpos_trajectory is not None:
                yield RobotQPosPath(qpos_trajectory)

    executor.register_function_implementation('get_all_blocking_grasping_objects', get_all_blocking_objects)
    executor.register_function_implementation('blocking_grasping', is_blocking)
    executor.register_function_implementation('get_all_blocking_placing_objects', get_all_blocking_objects)
    executor.register_function_implementation('blocking_placing', is_blocking)
    executor.register_function_implementation('on', on)

    executor.register_function_implementation('gen_grasp', sample_grasp)
    executor.register_function_implementation('gen_grasp_trajectory', sample_grasp_trajectory)
    executor.register_function_implementation('gen_placement', sample_placement)
    executor.register_function_implementation('gen_placement_on', sample_placement_on)
    executor.register_function_implementation('gen_placement_trajectory', sample_placement_trajectory)

    executor.register_function_implementation('gen_push', sample_push)
    executor.register_function_implementation('gen_push_trajectory', sample_push_trajectory)
    executor.register_function_implementation('gen_indirect_push', sample_indirect_push)
    executor.register_function_implementation('gen_indirect_push_trajectory', sample_indirect_push_trajectory)
