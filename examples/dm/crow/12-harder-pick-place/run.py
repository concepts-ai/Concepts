#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : run.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/31/2025
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp

import jacinle
import concepts.dm.crow as crow
from concepts.dm.crowhat.impl.pybullet.pybullet_sim_interfaces import PyBulletPhysicalControllerInterface, PyBulletSimulationControllerInterface
from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.pybullet.qddl_interface import PyBulletQDDLInterface
from concepts.simulator.pybullet.components.panda.panda_robot import PandaRobot
from concepts.simulator.pybullet.default_env import BulletEnvBase

from cogman import CognitionManager, load_skill_lib

crow.get_default_path_resolver().add_search_path(osp.dirname(__file__))

parser = jacinle.JacArgumentParser()
parser.add_argument('--domain', type=str, default='pick-place-domain.qddl')
parser.add_argument('--problem', type=str, default='pick-place-problem1.qddl')
args = parser.parse_args()


def build_env(title='(Physical World)'):
    client = BulletClient(is_gui=True, render_fps=120, additional_title=title)
    qddl_interface = PyBulletQDDLInterface(client)

    domain_qddl_filename = osp.join(osp.dirname(__file__), args.domain)
    problem_qddl_filename = osp.join(osp.dirname(__file__), args.problem)
    scene_metainfo, problem_metainfo = qddl_interface.load(domain_qddl_filename, problem_qddl_filename)

    robot = PandaRobot(client, client.world.body_names['panda'])
    robot.reset_home_qpos()  # ensure that the grippers are open
    return client, robot, scene_metainfo, problem_metainfo


def init_cogman_franka(libs, pb_client, pb_robot):
    domain = crow.load_domain_file('franka_base.cdl')
    cogman = CognitionManager(domain)
    cogman.executor = crow.CrowExecutor(domain)

    env = BulletEnvBase(pb_client)
    env.add_existing_robot(pb_robot)
    cogman.set_simulation_env(env)
    cogman.sci = PyBulletSimulationControllerInterface(env.client)
    cogman.sci.register_state_getter(lambda sci: cogman.get_state())

    cogman.pci = PyBulletSimulationControllerInterface(env.client)
    # cogman.pci = crow.CrowPhysicalControllerInterface(mock=True)

    load_skill_lib(cogman, 'franka_base', python_only=True)
    for lib in libs:
        load_skill_lib(cogman, lib)

    return cogman


def sync_cogman_object_list(cogman: CognitionManager):
    pb_env = cogman.get_simulation_env()
    cogman.clear_objects()
    for obj_index, obj_name in pb_env.world.body_names.int_to_string.items():
        if obj_name in ['panda', 'workspace']:
            continue

        labels = []
        if obj_name == 'table':
            labels = ['is_table']
        cogman.add_object(obj_name, obj_index, labels=labels)


def main():
    pb_client, pb_robot, scene_metainfo, problem_metainfo = build_env('(Planning World)')
    libs = ['franka_pick_place_push']
    cogman = init_cogman_franka(libs, pb_client, pb_robot)
    sync_cogman_object_list(cogman)

    plan = cogman.plan(problem_metainfo.goal, verbose=True, is_goal_serializable=False, is_goal_refinement_compressible=False)
    # plan = cogman.plan('not blocking_grasping(cube1, cube2)', verbose=True)

    print('Plan:')
    jacinle.stprint(plan[0].controller_actions)
    print('')
    pb_client.wait_for_user('Press Enter to execute the plan.')

    cogman.execute_plan(plan[0].controller_actions)
    pb_client.wait_for_user('Press Enter to exit.')


if __name__ == '__main__':
    main()
