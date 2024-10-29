#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : 2-solve-pick-place-hierarchy.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/22/2024
#
# This file is part of Project Concepts.

from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.pybullet.components.panda.panda_robot import PandaRobot
import concepts.dm.crow as crow
from concepts.benchmark.manip_tabletop.pick_place_hierarchy.pick_place_hierarchy import create_environment, get_available_tasks
from concepts.dm.crowhat.impl.pybullet.pybullet_sim_interfaces import PyBulletSimulationControllerInterface, PyBulletPhysicalControllerInterface, make_pybullet_remote_interfaces, \
    make_pybullet_simulator_ipc_ports
from concepts.dm.crowhat.impl.pybullet.pybullet_manipulator_interface import PyBulletSingleArmControllerInterface

DEFAULT_TCP_PORTS = (12020, 12021)


def build_env(task_identifier: str, title='(Physical World)', is_simulation: bool = False):
    client = BulletClient(is_gui=True, render_fps=120, additional_title=title, enable_realtime_rendering=False)
    _, scene_metainfo = create_environment(task_identifier, client)

    robot = PandaRobot(client, client.world.body_names['panda'])
    robot_controller_interface = PyBulletSingleArmControllerInterface(robot)
    robot.reset_home_qpos()

    scene_metainfo.robots.append(robot)

    if is_simulation:
        physical_controller_interface = PyBulletSimulationControllerInterface(client)
        robot_controller_interface.attach_simulation_interface(physical_controller_interface)
    else:
        physical_controller_interface = PyBulletPhysicalControllerInterface(client, dry_run=False)
        robot_controller_interface.attach_physical_interface(physical_controller_interface)

    return client, scene_metainfo, physical_controller_interface


def server(task_identifier: str, tcp_ports=DEFAULT_TCP_PORTS, ipc_ports=None):
    client, _, physical_controller_interface = build_env(task_identifier)
    physical_controller_interface.serve(tcp_ports=tcp_ports, ipc_ports=ipc_ports, redirect_ios=False)


def main(task_identifier: str, tcp_ports=DEFAULT_TCP_PORTS, ipc_ports=None):
    perception_interface, remote_physical_controller_interface = make_pybullet_remote_interfaces(tcp_ports=tcp_ports, ipc_ports=ipc_ports)
    client, scene_metainfo, local_physical_controller_interface = build_env(task_identifier, title='(Local World)', is_simulation=True)
    client.wait_for_duration(0.5)

    assert len(scene_metainfo.robots) == 1, 'Only one robot is supported.'
    robot = scene_metainfo.robots[0]

    from concepts.dm.crowhat.domains.pybullet_single_robot_rwi_pp_simplified import PyBulletSingleArmRWIPPSimplifiedDomain
    hat_domain = PyBulletSingleArmRWIPPSimplifiedDomain(robot, scene_metainfo)
    hat_domain.bind_perception_interface(perception_interface)
    hat_domain.bind_simulation_interface(local_physical_controller_interface)

    perception_interface.update_simulator()
    state = perception_interface.get_crow_state()
    print('Initial state:', state, flush=True)

    exeman = crow.CrowDefaultOpenLoopExecutionManager(
        hat_domain.executor, perception_interface, local_physical_controller_interface, remote_physical_controller_interface
    )
    goal = f'on(cube1, cube2)'
    exeman.run(goal)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=str, choices=['server', 'main', 'both'])
    parser.add_argument('--task', type=str, choices=get_available_tasks(), default='h0-simple')
    args = parser.parse_args()

    if args.target == 'server':
        server(args.task)
    elif args.target == 'main':
        main(args.task)
    elif args.target == 'both':
        tcp_ports = None
        ipc_ports = make_pybullet_simulator_ipc_ports()
        import multiprocessing as mp
        p1 = mp.Process(target=server, args=(args.task, tcp_ports, ipc_ports))
        p1.start()
        try:
            print('Server started. Waiting for 5 seconds...', flush=True)
            import time; time.sleep(5)
            print('Done.')
            main(args.task, tcp_ports, ipc_ports)
        except KeyboardInterrupt:
            pass
        except EOFError:
            pass
        except Exception as e:
            print(e)
        p1.terminate()
        p1.join()
