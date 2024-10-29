#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : bimanual_abc_pivot.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/07/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from concepts.benchmark.manip_tabletop.bimanual_abc.bimanual_abc_env_base import BimanualABCEnvBase


class BimanualABCPickupPlateEnv(BimanualABCEnvBase):
    def _reset_objects(self, metainfo: dict):
        plate_id = self.add_plate(0.6, (0.5, 0.0))
        self.metainfo['objects']['plate'] = {'id': plate_id, 'size': 0.3, 'position': (0.5, 0.0)}


class BimanualABCPivotBoxEnv(BimanualABCEnvBase):
    def _reset_objects(self, metainfo: dict):
        box_id = self.add_box((0.1, 0.15, 0.1), location_2d=(0.5, 0.0))
        self.metainfo['objects']['box'] = {'id': box_id}
