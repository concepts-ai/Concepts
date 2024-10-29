#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : paint_factory_policy.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/24/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from concepts.benchmark.manip_tabletop.paint_factory.paint_factory import PaintFactoryEnv


class PaintFactoryOraclePolicy(object):
    def __init__(self, env: PaintFactoryEnv):
        self._env = env

    @property
    def env(self):
        return self._env

    @property
    def block_info(self):
        return self.env.block_info

    @property
    def bowl_info(self):
        return self.env.bowl_info

    def act(self):
        """The oracle policy for the paint factory environment."""

        def paint(block_id, block_name, target_color):
            if 'dirty' in self.block_info[block_id]['color']:
                return dict(pick_object=block_name, place_object='clean-box')
            if self.block_info[block_id]['color'] == 'wet':
                return dict(pick_object=block_name, place_object='dry-box')

            # find a bowl with the target color
            for bowl_id, info in self.bowl_info.items():
                if info['color'] == target_color:
                    return dict(pick_object=block_name, place_object=info['name'])

            assert False, f'No bowl found for color {target_color}'

        def place(block_id, reference_id, relation):
            additional_place_offset = (0, 0)
            if relation == 'left of':
                additional_place_offset = (0, -0.05)
            elif relation == 'right of':
                additional_place_offset = (0, 0.05)
            return dict(pick_object=block_id, place_object=reference_id, additional_place_offset=additional_place_offset)

        for i, (block_id, info) in enumerate(self.block_info.items()):
            if info['color'] != self.env.color_goals[i]:
                return paint(block_id, info['name'], self.env.color_goals[i])

            # If the block is the first block, we just put it in the target box.
            if i == 0 and not self.env.is_relation_satisfied_in(block_id, self.env.target_box):
                return dict(pick_object=info['name'], place_object='target-box')

            for id1, id2, relation in self.env.relation_goals:
                if id1 == block_id:
                    if not self.env.is_relation_satisfied_in(id1, self.env.target_box) or not self.env.is_relation_satisfied(id1, id2, relation):
                        return place(info['name'], self.block_info[id2]['name'], relation)

        return None
