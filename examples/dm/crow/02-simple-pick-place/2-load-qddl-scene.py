#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : 2-load-qddl-scene.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/1/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp
from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.pybullet.components.panda.panda_robot import PandaRobot
import concepts.dm.crow as crow
from concepts.simulator.pybullet.qddl_interface import PyBulletQDDLInterface
from concepts.dm.crowhat.impl.pybullet.pybullet_sim_interfaces import PyBulletPhysicalControllerInterface, make_pybullet_remote_interfaces, make_pybullet_simulator_ipc_ports
from concepts.dm.crowhat.impl.pybullet.pybullet_manipulator_interface import PyBulletSingleArmControllerInterface

DEFAULT_TCP_PORTS = (12020, 12021)


def build_env(title='(Physical World)'):
    client = BulletClient(is_gui=True, render_fps=120, additional_title=title)
    qddl_interface = PyBulletQDDLInterface(client)

    domain_qddl_filename = osp.join(osp.dirname(__file__), 'simple-cube-in-box-domain.qddl')
    problem_qddl_filename = osp.join(osp.dirname(__file__), 'simple-cube-in-box-problem.qddl')
    qddl_interface.load_scene(domain_qddl_filename, problem_qddl_filename)

    robot = PandaRobot(client, client.world.body_names['panda'])
    robot_controller_interface = PyBulletSingleArmControllerInterface(robot)

    physical_controller_interface = PyBulletPhysicalControllerInterface(client)
    robot_controller_interface.attach_physical_interface(physical_controller_interface)

    return client, physical_controller_interface


def server(tcp_ports=DEFAULT_TCP_PORTS, ipc_ports=None):
    client, physical_controller_interface = build_env()
    physical_controller_interface.serve(tcp_ports=tcp_ports, ipc_ports=ipc_ports, redirect_stderr='/tmp/pybullet-server.log')


def main(tcp_ports=DEFAULT_TCP_PORTS, ipc_ports=None):
    perception_interface, remote_physical_controller_interface = make_pybullet_remote_interfaces(tcp_ports=tcp_ports, ipc_ports=ipc_ports)
    client, local_physical_controller_interface = build_env(title='(Local World)')
    client.wait_for_duration(0.5)

    print('Available commands:')
    print('scene: get the current scene. It will print a relational state representation.')
    print('sync: sync the current scene with the perception interface.')
    print('rhome: move the robot to the home pose in the remote controller.')
    print('rmove <x> <y> <z>: move the robot to the specified position in the remote controller.')
    print('lhome: move the robot to the home pose in the local controller.')
    print('lmove <x> <y> <z>: move the robot to the specified position in the local controller.')
    print('exit: exit the program.')

    while True:
        command = input('Enter a command: ')
        if command == 'scene':
            rv = perception_interface.get_scene()
            print(str(rv))
            continue
        elif command == 'sync':
            rv = perception_interface.get_scene()
            rv.restore()
            client.wait_for_duration(0.5)
            continue
        elif command == 'rhome':
            controller = crow.CrowController('move_home', [])
            rv = remote_physical_controller_interface.step(crow.CrowControllerApplier(controller, []))
        elif command.startswith('rmove'):
            try:
                x, y, z = map(float, command.split()[1:])
            except:
                print(f'Invalid command: {command}. Expected: rmove <x> <y> <z>.')
                continue
            controller = crow.CrowController('move_pose', [])
            rv = remote_physical_controller_interface.step(crow.CrowControllerApplier(controller, [[x, y, z], [1, 0, 0, 0]]))
        elif command == 'lhome':
            controller = crow.CrowController('move_home', [])
            try:
                rv = local_physical_controller_interface.step(crow.CrowControllerApplier(controller, []))
            except Exception as e:
                print(e)
                continue
        elif command.startswith('lmove'):
            try:
                x, y, z = map(float, command.split()[1:])
            except:
                print(f'Invalid command: {command}. Expected: lmove <x> <y> <z>.')
                continue
            controller = crow.CrowController('move_pose', [])
            try:
                rv = local_physical_controller_interface.step(crow.CrowControllerApplier(controller, [[x, y, z], [1, 0, 0, 0]]))
            except Exception as e:
                print(e)
                continue
        elif command == 'exit':
            break
        else:
            print('Unknown command.')
            continue
        print(rv)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=str, choices=['server', 'main', 'both'])
    args = parser.parse_args()

    if args.target == 'server':
        server()
    elif args.target == 'main':
        main()
    elif args.target == 'both':
        tcp_ports = None
        ipc_ports = make_pybullet_simulator_ipc_ports()
        import multiprocessing as mp
        p1 = mp.Process(target=server, args=(tcp_ports, ipc_ports))
        p1.start()
        try:
            print('Server started. Waiting for 5 seconds...', flush=True)
            import time; time.sleep(5)
            print('Done.')
            main(tcp_ports, ipc_ports)
        except KeyboardInterrupt:
            pass
        except EOFError:
            pass
        except Exception as e:
            print(e)
        p1.terminate()
        p1.join()
