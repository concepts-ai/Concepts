#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : 3-control-panda.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/22/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.pybullet.components.panda.panda_robot import PandaRobot
from concepts.simulator.pybullet.qddl_interface import PyBulletQDDLInterface
from concepts.dm.crowhat.impl.pybullet.pybullet_sim_interfaces import PyBulletSimulationControllerInterface
from concepts.dm.crowhat.impl.pybullet.pybullet_manipulator_interface import PyBulletSingleArmControllerInterface


def build_env():
    client = BulletClient(is_gui=True, render_fps=120, fps=60)
    qddl_interface = PyBulletQDDLInterface(client)

    domain_qddl_filename = 'simple-cube-in-box-domain.qddl'
    problem_qddl_filename = 'simple-cube-in-box-problem.qddl'

    qddl_interface.load_scene(domain_qddl_filename, problem_qddl_filename)

    robot = PandaRobot(client, client.world.body_names['panda'])
    control_interface = PyBulletSingleArmControllerInterface(robot)
    robot.reset_home_qpos()

    sim_controller_interface = PyBulletSimulationControllerInterface(client)
    control_interface.attach_simulation_interface(sim_controller_interface)

    return client, sim_controller_interface


def main():
    client, sim = build_env()
    client.wait_for_user('Press Enter to start the simulation.')

    # Test the robot primitive controllers and restore_context.
    with sim.restore_context():
        sim.step_internal('move_pose', (0.5, 0.3, 0.1), (0, 1, 0, 0))
        sim.step_internal('move_pose', (0.5, 0.3, 0.05), (0, 1, 0, 0))
        sim.step_internal('grasp')
        with sim.restore_context():
            sim.step_internal('move_home')
        sim.step_internal('move_home')

    with sim.restore_context():
        sim.step_internal('move_pose', (0.5, -0.3, 0.1), (0, 1, 0, 0))
        sim.step_internal('move_pose', (0.5, -0.3, 0.05), (0, 1, 0, 0))
        sim.step_internal('move_home')

    client.wait_for_user('Press Enter to exit.')


if __name__ == '__main__':
    main()
