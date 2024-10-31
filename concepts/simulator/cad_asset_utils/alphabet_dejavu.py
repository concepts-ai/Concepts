#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : alphabet_arial.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/01/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp
from typing import Optional

from concepts.simulator.urdf_utils.obj2urdf import ObjectUrdfBuilder
from concepts.simulator.cad_asset_utils.assets_base import CADAssetCollection


class AlphabetDejavuCollection(CADAssetCollection):
    """A collection of object URDFs for the Arial alphabet."""

    def __init__(self, asset_root: Optional[str] = None, use_vhacd: bool = False):
        if asset_root is None:
            asset_root = self.get_default_asset_root()

        self.root = asset_root

        self.available_models = dict()
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            self.available_models[letter] = osp.join(self.root, f'{letter}', f'{letter}.obj')
        # for letter in 'abcdefghijklmnopqrstuvwxyz':
            self.available_models[letter] = osp.join(self.root, f'{letter}', f'{letter}.obj')

        self.use_vhacd = use_vhacd

    def get_default_asset_root(self) -> str:
        return self.get_assets_path('objects', 'alphabet_dejavu')

    def build_urdf(self, object_path) -> None:
        object_dir = osp.dirname(object_path)
        object_builder = ObjectUrdfBuilder(object_dir)
        object_builder.build_urdf(object_path, force_overwrite=True, decompose_concave=True, force_decompose=False, center='mass')

    def assert_build_urdf(self, identifier) -> str:
        assert identifier in self.available_models, f'Unknown identifier: {identifier}'

        obj_path = self.available_models[identifier]
        urdf_path = obj_path + '.urdf'

        if not osp.isfile(urdf_path):
            self.build_urdf(obj_path)

        return urdf_path

    def get_urdf(self, identifier: str) -> str:
        return self.assert_build_urdf(identifier)

    def get_obj_filename(self, identifier: str, vhacd: Optional[bool] = None) -> str:
        assert identifier in self.available_models, f'Unknown identifier: {identifier}'
        if vhacd is None:
            vhacd = self.use_vhacd
        if vhacd:
            return self.available_models[identifier].replace('.obj', '_vhacd.obj')
        return self.available_models[identifier]
