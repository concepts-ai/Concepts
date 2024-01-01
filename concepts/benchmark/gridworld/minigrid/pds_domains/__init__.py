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

__all__ = ['get_minigrid_domain_filename']


def get_minigrid_domain_filename(encoding: str = 'full') -> str:
    """Get the domain filename of the crafting world."""
    return osp.join(osp.dirname(__file__), f'minigrid-domain-v20220407-{encoding}.pdsketch')
