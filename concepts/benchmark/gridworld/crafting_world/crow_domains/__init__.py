#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/05/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp

__all__ = ['get_crafting_world_domain_filename']


def get_crafting_world_domain_filename(station_agnostic: bool = True, regenerate: bool = False) -> str:
    """Get the domain filename of the crafting world (teleport)."""

    if regenerate:
        from .cdl_gen import main_station_agnostic
        main_station_agnostic()

    if station_agnostic:
        return osp.join(osp.dirname(__file__), 'crafting_world_station_agnostic.cdl')
    else:
        return osp.join(osp.dirname(__file__), 'crafting_world.cdl')

