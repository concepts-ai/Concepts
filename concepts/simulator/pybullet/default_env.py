#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : default_env.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/08/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp
from typing import Optional, Any, List, Dict

import numpy as np
import jacinle.io as io
from jacinle.logging import get_logger

from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.pybullet.components.asset_libraries.shapenet import ShapeNetCoreLoader
from concepts.utils.typing_utils import Vec3f, Vec4f

logger = get_logger(__file__)


class BulletEnvBase(object):
    def __init__(self, client: Optional[BulletClient] = None, np_random: Optional[np.random.RandomState] = None, seed: Optional[int] = None, is_gui: bool = True):
        if client is None:
            client = BulletClient(is_gui=is_gui)
        self._client = client

        if np_random is None:
            if seed is None:
                self._np_random = np.random.RandomState()
            else:
                self._np_random = np.random.RandomState(seed)
        else:
            self._np_random = np_random

        self.robot = None
        self.robots = list()

    def add_existing_robot(self, robot: Any):
        self.robot = robot
        self.robots.append(robot)

    @property
    def client(self) -> BulletClient:
        return self._client

    @property
    def world(self):
        return self.client.world

    @property
    def w(self):
        return self.client.world

    @property
    def p(self):
        return self.client.p

    @property
    def np_random(self) -> np.random.RandomState:
        return self._np_random

    def seed(self, seed: int):
        self.np_random.seed(seed)

    @property
    def ocrtoc_root(self) -> str:
        return osp.join(self.client.assets_root, 'objects', 'ocrtoc')

    def list_ocrtoc_objects(self) -> List[str]:
        all_model_names = list()
        for dirname in io.lsdir(self.ocrtoc_root, '*'):
            if osp.isdir(dirname) and osp.exists(osp.join(dirname, 'model.urdf')):
                all_model_names.append(osp.basename(dirname))
        return all_model_names

    def add_ocrtoc_object(
        self, identifier: str, location_3d: Vec3f, scale: float = 1.0, name: Optional[str] = None, *,
        static: bool = False, quat: Vec4f = (0, 0, 0, 1)
    ):
        if name is None:
            name = identifier
        return self.client.load_urdf(
            osp.join(self.ocrtoc_root, identifier, 'model.urdf'),
            location_3d,
            quat,
            body_name=name,
            scale=scale,
            static=static,
        )

    @property
    def ycb_simple_root(self) -> str:
        return osp.join(self.client.assets_root, 'objects', 'ycb_simplified')

    def list_ycb_simplified_objects(self) -> List[str]:
        all_model_names = list()
        for dirname in io.lsdir(self.ycb_simple_root, '*'):
            if osp.isdir(dirname) and osp.exists(osp.join(dirname, 'textured_simple.obj.urdf')):
                all_model_names.append(osp.basename(dirname))
        return all_model_names

    def add_ycb_simplified_object(
        self, identifier: str, location_3d: Vec3f, scale: float = 1.0, name: Optional[str] = None, *,
        static: bool = False, quat: Vec4f = (0, 0, 0, 1)
    ):
        if name is None:
            name = identifier
        return self.client.load_urdf(
            osp.join(self.ycb_simple_root, identifier, 'textured_simple.obj.urdf'),
            location_3d,
            quat,
            body_name=name,
            scale=scale,
            static=static,
        )

    _shapenet_core_loader: Optional[ShapeNetCoreLoader] = None

    @property
    def shapenet_core_loader(self) -> ShapeNetCoreLoader:
        return self._shapenet_core_loader

    def initialize_shapenet_core_loader(self, shapenet_dir: str):
        if self._shapenet_core_loader is not None:
            logger.warning('ShapeNetCoreLoader is already initialized. Ignored.')
            return
        self._shapenet_core_loader = ShapeNetCoreLoader(self.client, shapenet_dir)

    def list_shape_net_core_objects(self) -> Dict[str, Dict[str, Any]]:
        assert self._shapenet_core_loader is not None, 'ShapeNetCoreLoader is not initialized.'
        return self._shapenet_core_loader.available_models

    def add_shape_net_core_object(
        self, synset_identifier: str, model_identifier: str, location_3d: Vec3f, scale: float = 1.0, name: Optional[str] = None, *,
        static: bool = False, quat: Vec4f = (0, 0, 0, 1)
    ):
        assert self._shapenet_core_loader is not None, 'ShapeNetCoreLoader is not initialized.'
        return self._shapenet_core_loader.load_urdf(synset_identifier, model_identifier, location_3d, scale=scale, name=name, static=static, quat=quat)

    @property
    def igibson_root(self) -> str:
        return osp.join(self.client.assets_root, 'igibson')

    def find_igibson_object_by_category(self, category: str) -> List[str]:
        path = osp.join(self.client.assets_root, 'igibson', 'objects', category)
        all_files = io.lsdir(path, '*/*.urdf')
        assert len(all_files) > 0, f'No urdf files found in {path}.'
        return all_files

    def add_igibson_object_by_category(
        self, category: str, location_3d: Vec3f, scale: float = 1.0, name: Optional[str] = None, *,
        urdf_file: Optional[str] = None,
        static: bool = False, quat: Vec4f = (0, 0, 0, 1),
    ):
        if name is None:
            name = category

        if urdf_file is None:
            all_files = self.find_igibson_object_by_category(category)
            urdf_file = all_files[0]

        return self.client.load_urdf(
            urdf_file,
            location_3d,
            quat,
            body_name=name,
            scale=scale,
            static=static,
        )

