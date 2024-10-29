#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pybullet_tabletop_pdsinterface.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/03/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp
from typing import Optional

from jacinle.logging import get_logger
from concepts.dm.pdsketch.domain import State
from concepts.dm.pdsketch.executor import PDSketchExecutor
from concepts.benchmark.manip_tabletop.pybullet_tabletop_base.pybullet_tabletop import TableTopEnv

logger = get_logger(__file__)

__all__ = ['get_tabletop_base_domain_filename', 'PybulletTableTopPDSketchInterface']


def get_tabletop_base_domain_filename() -> str:
    return osp.join(osp.dirname(__file__), 'pybullet_tabletop_base.pdsketch')


class PybulletTableTopPDSketchInterface(object):
    def __init__(self, env: TableTopEnv, executor: Optional[PDSketchExecutor] = None):
        self._executor = executor
        self._env = env

        if not hasattr(env.robot, 'gripper_constraint'):
            logger.warning('The robot does not have a gripper constraint. The interface may not work properly.')

    @property
    def executor(self) -> PDSketchExecutor:
        if self._executor is None:
            raise RuntimeError('Executor is not initialized yet.')
        return self._executor

    def set_executor(self, executor: PDSketchExecutor):
        self._executor = executor

    @property
    def env(self) -> TableTopEnv:
        return self._env

    def get_pds_state(self) -> State:
        objects = dict()
        for name, info in self.env.metainfo.items():
            object_type = self.executor.domain.types['robot'] if name == 'robot' else self.executor.domain.types['item']
            objects[name] = object_type

        state, ctx = self.executor.new_state(objects, create_context=True)

        for name, info in self.env.metainfo.items():
            index = info['id']
            if name == 'robot':
                ctx.set_value('robot-qpos', [name], self.env.robot.get_qpos())
                ctx.set_value('robot-identifier', [name], index)
            else:
                ctx.set_value('item-pose', [name], self.env.world.get_body_state_by_id(index).get_7dpose())
                ctx.set_value('item-identifier', [name], index)

        for name, info in self.env.metainfo.items():
            if name not in ('robot', 'table', 'panda'):
                for name2 in self.env.get_support(info['id']):
                    if name2 not in ('robot', 'panda'):
                        ctx.set_value('support', [name, name2], True)

        ctx.init_feature('moveable')
        for name, info in self.env.metainfo.items():
            if 'moveable' in info and info['moveable']:
                ctx.set_value('moveable', [name], True)

        if hasattr(self.env.robot, 'gripper_constraint'):
            if self.env.robot.gripper_constraint is None:
                ctx.define_predicates([ctx.robot_hands_free('robot')])
            else:
                constraint = self.env.world.get_constraint(self.env.robot.gripper_constraint)
                name = self.env.world.body_names[constraint.child_body]
                ctx.define_predicates([ctx.robot_holding_item('robot', name)])

        return state
