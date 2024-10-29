#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : shapenet.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/01/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp
import jacinle.io as io

from jacinle.logging import get_logger
from concepts.simulator.cad_asset_utils.assets_base import CADAssetCollection

logger = get_logger(__file__)


class ShapeNetCoreCollection(CADAssetCollection):
    def __init__(self, shapenet_root: str):
        self.shapenet_root = shapenet_root
        self.available_models = self.load_available_models()

    def load_available_models(self):
        taxonomy_file = osp.join(self.shapenet_root, 'taxonomy.json')
        taxonomy_data = io.load(taxonomy_file)
        all_model_names = dict()
        for synset in taxonomy_data:
            synset_id = synset['synsetId']
            if osp.isdir(osp.join(self.shapenet_root, synset_id)):
                identifier = synset['name'].split(',')[0]
                all_model_names[identifier] = dict(synset_id=synset_id, full_name=synset['name'], available_models=list())
                for model_dir in io.lsdir(osp.join(self.shapenet_root, synset_id), '*'):
                    if osp.isdir(model_dir):
                        all_model_names[identifier]['available_models'].append(osp.basename(model_dir))

        return all_model_names

    def build_urdf(self, model_dir: str, urdf_dir: str, identifier: str, model_identifier: str) -> str:
        """Build URDF for the specific model."""

        io.mkdir(urdf_dir)
        for file in io.lsdir(model_dir, '*', return_type='full'):
            if osp.isfile(file):
                io.link(file, osp.join(urdf_dir, osp.basename(file)), use_relative_path=True)

        if not osp.isfile(osp.join(urdf_dir, 'model_normalized.obj')):
            raise FileExistsError(f'No normalized obj found for {identifier}/{model_identifier}')

        object_builder = self.make_urdf_builder(urdf_dir)
        object_builder.build_urdf(osp.join(urdf_dir, 'model_normalized.obj'), force_overwrite=True, decompose_concave=True, force_decompose=False, center='mass')
        return osp.join(urdf_dir, 'model_normalized.obj.urdf')

    def get_urdf(self, identifier: str, model_identifier: str) -> str:
        synset_id = self.available_models[identifier]['synset_id']
        asset_dir = osp.join(self.shapenet_root, synset_id, model_identifier)
        model_dir = osp.join(asset_dir, 'models')
        urdf_dir = osp.join(asset_dir, 'urdf')

        if osp.isdir(urdf_dir):
            if osp.isfile(osp.join(urdf_dir, 'model_normalized.obj.urdf')):
                return osp.join(urdf_dir, 'model_normalized.obj.urdf')

            logger.warning(f'No normalized urdf found for {identifier}/{model_identifier}, re-building one...')
            io.remove(osp.join(model_dir, 'urdf'))

        return self.build_urdf(model_dir, urdf_dir, identifier, model_identifier)
