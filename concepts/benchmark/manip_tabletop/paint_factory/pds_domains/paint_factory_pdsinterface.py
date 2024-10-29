#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : paint_factory_pdsinterface.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/03/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp
from typing import Optional

import cv2
import torch
import numpy as np

from concepts.dm.pdsketch.executor import PDSketchExecutor
from concepts.dm.pdsketch.domain import State
from concepts.benchmark.manip_tabletop.paint_factory.paint_factory import PaintFactoryEnv

__all__ = ['get_paint_factory_domain_filename', 'PaintFactoryPDSketchInterface']


def get_paint_factory_domain_filename() -> str:
    """Get the domain filename of the crafting world."""
    return osp.join(osp.dirname(__file__), f'paint-factory-domain-v20231225.pdsketch')


class PaintFactoryPDSketchInterface(object):
    def __init__(self, env: PaintFactoryEnv, executor: Optional[PDSketchExecutor] = None):
        self._executor = executor
        self._env = env

    @property
    def executor(self) -> PDSketchExecutor:
        if self._executor is None:
            raise RuntimeError('Executor is not initialized yet.')
        return self._executor

    def set_executor(self, executor: PDSketchExecutor):
        self._executor = executor

    @property
    def env(self) -> PaintFactoryEnv:
        return self._env

    def get_pds_state(self) -> State:
        t_item = self.executor.domain.types['item']
        objects = dict()

        objects['robot'] = self.executor.domain.types['robot']
        for _, item_info in self.env.iter_objects():
            objects[item_info['name']] = t_item

        state, ctx = self.executor.new_state(objects, create_context=True)

        ctx.define_feature('robot-qpos', torch.tensor([self.env.robot.get_qpos()], dtype=torch.float32))
        ctx.set_value('robot-identifier', ['robot'], self.env.robot.get_body_id())

        for item_id, item_info in self.env.iter_objects():
            ctx.set_value('item-pose', [item_info['name']], self.env.world.get_body_state_by_id(item_id).pos[:2])  # Only use (x, y).
            ctx.set_value('item-identifier', [item_info['name']], item_id)

            if item_info['name'].startswith('block'):
                ctx.set_value('is-block', [item_info['name']], True)
            elif item_info['name'].startswith('bowl'):
                ctx.set_value('is-machine', [item_info['name']], True)
            elif item_info['name'] == 'target-box':
                ctx.set_value('is-target', [item_info['name']], True)
            elif item_info['name'] in ('clean-box', 'dry-box'):
                ctx.set_value('is-machine', [item_info['name']], True)
            else:
                raise ValueError('Unknown object name: {}'.format(item_info['name']))

        img, depth, segm = self.env.world.render_image(self.env.topdown_camera, normalize_depth=True)

        tensors = list()
        for item_id, item_info in self.env.iter_objects():
            item_segm = (segm == item_id)
            # Use OpenCV instead to get the bounding box
            bounding_box = cv2.boundingRect(item_segm.astype(np.uint8))  # (x, y, w, h)
            bounding_box = (bounding_box[0], bounding_box[1], bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3])
            # Expand the bounding box by 5 pixels and clip to image size
            bounding_box = (
                max(0, bounding_box[0] - 5),
                max(0, bounding_box[1] - 5),
                min(img.shape[1], bounding_box[2] + 5),
                min(img.shape[0], bounding_box[3] + 5),
            )
            item_img = img[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]
            item_img = cv2.resize(item_img, (32, 32), interpolation=cv2.INTER_AREA)
            item_img = item_img.flatten()
            tensors.append(torch.tensor(item_img, dtype=torch.float32))

        ctx.define_feature('item-image', torch.stack(tensors, dim=0))

        # TODO(Jiayuan Mao @ 2024/01/24): add gripper constraint.
        # if env.robot.gripper_constraint is None:
        #     ctx.define_predicates([ctx.robot_hands_free('robot')])
        # else:
        #     constraint = env.world.get_constraint(env.robot.gripper_constraint)
        #     name = env.world.body_names[constraint.child_body]
        #     ctx.define_predicates([ctx.robot_holding_item('robot', name)])

        return state
