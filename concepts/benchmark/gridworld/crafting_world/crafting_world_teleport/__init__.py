#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/06/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp

__all__ = ['get_domain_filename', 'get_station_agnostic_domain_filename']


def get_domain_filename() -> str:
    """Get the domain filename of the crafting world (teleport)."""
    return osp.join(osp.dirname(__file__), 'domain.pddl')


def get_station_agnostic_domain_filename() -> str:
    """Get the domain filename of the crafting world (teleport) with station-agnostic actions."""
    return osp.join(osp.dirname(__file__), 'domain-station-agnostic.pddl')


def _check_file_exists():
    if not osp.isfile(get_domain_filename()) or not osp.isfile(get_station_agnostic_domain_filename()):
        import concepts.benchmark.gridworld.crafting_world.crafting_world_teleport.domain_gen as gen
        print('Generating the domain files for the crafting world (teleport)...')
        gen.main()
        gen.main_station_agnostic()


_check_file_exists()