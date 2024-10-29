#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pick_place_hierarchy.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/07/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp
from typing import Optional, Tuple

from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.pybullet.qddl_interface import PyBulletQDDLInterface, QDDLSceneMetainfo


def get_available_tasks():
    return [
        'h0-simple',
        'h1-pick-blocker',
        'h1-place-blocker',
        'h1-pick-regrasp',
        'h2-pick-place-blocker',
    ]


def get_qddl_domain_filename():
    return osp.join(osp.dirname(__file__), 'qddl_files', 'pick-place-hierarchy-domain.qddl')


def get_qddl_problem_filename(identifier: str):
    return osp.join(osp.dirname(__file__), 'qddl_files', f'{identifier}.qddl')


def create_environment(task_identifier: str, client: Optional[BulletClient] = None) -> Tuple[BulletClient, QDDLSceneMetainfo]:
    if client is None:
        client = BulletClient(is_gui=True)

    interface = PyBulletQDDLInterface(client)
    metainfo = interface.load_scene(
        get_qddl_domain_filename(),
        get_qddl_problem_filename(task_identifier)
    )

    return client, metainfo
