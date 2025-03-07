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
from concepts.simulator.pybullet.qddl_interface import PyBulletQDDLInterface


def build_env(args):
    client = BulletClient(is_gui=True, render_fps=120, additional_title=args.title)
    qddl_interface = PyBulletQDDLInterface(client)

    domain_qddl_filename = args.domain
    problem_qddl_filename = args.problem
    qddl_interface.load_scene(domain_qddl_filename, problem_qddl_filename)

    client.wait_forever()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('domain', type=str)
    parser.add_argument('problem', type=str)
    parser.add_argument('--title', type=str, default='QDDL Scene')
    args = parser.parse_args()

    build_env(args)

