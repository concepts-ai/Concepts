#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : paint_factory_crow_interface.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/03/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import cv2
import torch
import numpy as np

from concepts.benchmark.manip_tabletop.paint_factory.paint_factory import PaintFactoryEnv
from concepts.dm.crow.interfaces.perception_interface import CrowPerceptionInterface
from concepts.dm.crow.crow_domain import CrowDomain, CrowState

__all__ = ['PaintFactoryPerceptionInterface']


class PaintFactoryPerceptionInterface(CrowPerceptionInterface):
    def __init__(self, env: PaintFactoryEnv, domain: CrowDomain):
        super().__init__()

        self.env = env
        self.domain = domain

    def get_crow_state(self) -> CrowState:
        """Get the state of the perception interface."""
        objects = dict()

        objects['robot'] = 'Robot'
        for _, item_info in self.env.iter_objects():
            objects[item_info['name']] = 'Object'

        state = CrowState.make_empty_state(self.domain, objects)

        state.fast_set_value('robot_identifier', ['robot'], self.env.robot.get_body_id())

        for item_id, item_info in self.env.iter_objects():
            state.fast_set_value('object_pose', [item_info['name']], torch.tensor(self.env.world.get_body_state_by_id(item_id).pos[:2]))  # Only use (x, y).
            state.fast_set_value('object_identifier', [item_info['name']], item_id)

            if item_info['name'].startswith('block'):
                state.fast_set_value('is_block', [item_info['name']], True)
            elif item_info['name'].startswith('bowl'):
                state.fast_set_value('is_machine', [item_info['name']], True)
            elif item_info['name'] == 'target-box':
                state.fast_set_value('is_target', [item_info['name']], True)
            elif item_info['name'] in ('clean-box', 'dry-box'):
                state.fast_set_value('is_machine', [item_info['name']], True)
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

        state.batch_set_value('object_image', torch.stack(tensors, dim=0))

        return state
