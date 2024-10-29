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


def get_paint_factory_domain_filename():
    return osp.join(osp.dirname(__file__), 'paint-factory-v20231225.cdl')

