#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : bimanual_abc_lift.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/15/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional
from concepts.math.rotationlib_xyzw import rpy, quat_mul
from concepts.simulator.pybullet.client import BulletClient
from concepts.benchmark.manip_tabletop.bimanual_abc.bimanual_abc_env_base import BimanualABCEnvBase

__all__ = [
    'BimanualABCLiftObjectEnvBase', 'BimanualABCLiftBoxEnv',
    'BimanualABCLiftTEnv',
    'BimanualABCLiftArialTEnv', 'BimanualABCLiftArialAnythingEnv'
]


class BimanualABCLiftObjectEnvBase(BimanualABCEnvBase):
    def __init__(self, client: Optional[BulletClient] = None, is_gui: bool = True, seed: int = 1234, add_tools: bool = True):
        super().__init__(client, is_gui, seed)
        self.add_tools = add_tools

    def _get_lifting_bar_length(self) -> float:
        return 0.15

    def _reset_objects(self, metainfo: dict):
        if self.add_tools:
            lifting_bar_length = self._get_lifting_bar_length()
            box_id1 = self.add_box((0.025, lifting_bar_length, 0.025), location_2d=(0.35, -.2), color_rgba=(0, 1, 0, 1), name='box1')
            box_id2 = self.add_box((0.025, lifting_bar_length, 0.025), location_2d=(0.65, -.2), color_rgba=(0, 0, 1, 1), name='box2')
            metainfo['objects']['box1'] = {'id': box_id1}
            metainfo['objects']['box2'] = {'id': box_id2}


class BimanualABCLiftBoxEnv(BimanualABCLiftObjectEnvBase):
    def _reset_objects(self, metainfo: dict):
        super()._reset_objects(metainfo)
        box_id = self.add_box((0.1, 0.2, 0.1), location_2d=(0.5, 0.0))
        metainfo['objects']['box'] = {'id': box_id}


class BimanualABCLiftTEnv(BimanualABCLiftObjectEnvBase):
    def _reset_objects(self, metainfo: dict):
        super()._reset_objects(metainfo)

        T_id = self.add_t_shape(
            size1_2d=(0.2, 0.1),
            size2_2d=(0.05, 0.4),
            thickness=0.05,
            pos=(0.5, 0.0, 0.1),  # The center of the shape is at the center of the handle.
            quat=rpy(0, -90, 0),
        )
        metainfo['objects']['T'] = {'id': T_id}


class BimanualABCLiftArialTEnv(BimanualABCLiftObjectEnvBase):
    def _get_lifting_bar_length(self) -> float:
        return 0.25

    def _reset_objects(self, metainfo: dict):
        super()._reset_objects(metainfo)

        default_quat = quat_mul(rpy(0, 0, 90), rpy(0, 180, 0))

        T_id = self.add_alphabet_arial_object('T', location_3d=(0.5, 0.0, 0.1), quat=quat_mul(default_quat, rpy(0, -45, 0)), scale=1.5)
        metainfo['objects']['T'] = {'id': T_id}


class BimanualABCLiftArialAnythingEnv(BimanualABCLiftObjectEnvBase):
    def __init__(self, alphabet, client: Optional[BulletClient] = None, is_gui: bool = True, seed: int = 1234):
        super().__init__(client, is_gui, seed)
        self.alphabet = alphabet

        assert len(self.alphabet) == 1 and self.alphabet.isalpha()

    def _get_lifting_bar_length(self) -> float:
        return 0.25

    def _reset_objects(self, metainfo: dict):
        super()._reset_objects(metainfo)

        default_quat = quat_mul(rpy(0, 0, 90), rpy(0, 180, 0))

        letter_id = self.add_alphabet_arial_object(self.alphabet, location_3d=(0.5, 0.0, 0.1), quat=quat_mul(default_quat, rpy(0, -45, 0)), scale=1.5)
        metainfo['objects']['letter'] = {'id': letter_id}
