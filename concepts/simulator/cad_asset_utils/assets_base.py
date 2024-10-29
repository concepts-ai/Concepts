#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : assets_base.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/01/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp

from concepts.simulator.urdf_utils.obj2urdf import ObjectUrdfBuilder


class CADAssetCollection(object):
    def get_concepts_assets_path(self) -> str:
        return osp.abspath(osp.join(osp.dirname(__file__), '..', '..', 'assets'))

    def get_assets_path(self, *subpath) -> str:
        return osp.join(self.get_concepts_assets_path(), *subpath)

    def make_urdf_builder(self, object_dir: str) -> ObjectUrdfBuilder:
        return ObjectUrdfBuilder(object_dir)
