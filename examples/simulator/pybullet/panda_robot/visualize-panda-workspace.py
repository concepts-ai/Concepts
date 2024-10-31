#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : visualize-panda-workspace.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/11/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from concepts.simulator.pybullet.client import BulletClient
from concepts.benchmark.manip_tabletop.pybullet_tabletop_base.pybullet_tabletop import TableTopEnv
from concepts.simulator.pybullet.components.panda.panda_robot import PandaRobot


def main():
    bclient = BulletClient(is_gui=True)
    env = TableTopEnv(bclient)
    env.add_workspace_boundary((-0.2, 2.0), (-1.0, 1.0), (-0.1, 1.5))
    env.add_robot('panda', robot_kwargs={'version': 'soft_finger'})

    print(env.world.link_names.int_to_string)

    bclient.wait_forever()


if __name__ == '__main__':
    main()
