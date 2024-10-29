#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : bimanual_abc_env_base.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/07/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional

from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.cad_asset_utils.alphabet_arial import AlphabetArialCollection
from concepts.benchmark.manip_tabletop.pybullet_tabletop_base.pybullet_tabletop import TableTopEnv

__all__ = ['BimanualABCEnvBase']

from concepts.utils.typing_utils import Vec3f, Vec4f


class BimanualABCEnvBase(TableTopEnv):
    def __init__(self, client: Optional[BulletClient] = None, is_gui: bool = True, seed: int = 1234):
        super().__init__(client, is_gui=is_gui, seed=seed)
        self.alphabet_arial_collection = AlphabetArialCollection()

    def reset(self):
        """Reset the environment. This function will create a metainfo dict and call _reset_objects to populate the scene."""
        self.metainfo = {'objects': dict()}
        with self.client.disable_rendering(suppress_stdout=True):
            table_id = self.add_workspace(large=True)
            self.metainfo['objects']['table'] = {'id': table_id}
            self.add_robot('panda', (0, 0, 0), name='panda1')
            self.add_robot('panda', (0.5, -0.8, 0), (0, 0, 0.707, 0.707), name='panda2')
            self.set_default_debug_camera()
            self._reset_objects(self.metainfo)

    default_alphabet_arial_scale = 0.01

    def add_alphabet_arial_object(
        self, identifier: str, location_3d: Vec3f, scale: float = 1.0, name: Optional[str] = None, *,
        static: bool = False, quat: Vec4f = (0, 0, 0, 1)
    ):
        scale = scale * self.default_alphabet_arial_scale
        urdf_path = self.alphabet_arial_collection.get_urdf(identifier)
        return self.client.load_urdf(urdf_path, location_3d, quat, scale=scale, body_name=name, static=static)

    def _reset_objects(self, metainfo: dict):
        """Populate the scene with objects. This function should be implemented by subclasses.

        Args:
            metainfo (dict): the metainfo dict to be populated. Subclasses should add their objects to this dict in-place.
        """
        raise NotImplementedError()
