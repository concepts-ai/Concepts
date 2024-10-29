#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : shapenet.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/10/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional

from concepts.simulator.cad_asset_utils.shapenet import ShapeNetCoreCollection
from concepts.simulator.pybullet.client import BulletClient
from concepts.utils.typing_utils import Vec3f, Vec4f


class ShapeNetCoreLoader(object):
    def __init__(self, client: BulletClient, shapenet_root: str):
        self.client = client
        self.asset_collection = ShapeNetCoreCollection(shapenet_root)

    def load_urdf(
        self, synset_identifier: str, model_identifier: str, location_3d: Vec3f, scale: float = 1.0, name: Optional[str] = None, *,
        static: bool = False, quat: Vec4f = (0, 0, 0, 1)
    ):
        urdf_file = self.asset_collection.get_urdf(synset_identifier, model_identifier)
        if name is None:
            name = synset_identifier

        return self.client.load_urdf(urdf_file, location_3d, quat, body_name=name, scale=scale, static=static)

