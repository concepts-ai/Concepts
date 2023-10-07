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

__all__ = ['get_domain_filename']


def get_domain_filename():
    return osp.join(osp.dirname(__file__), 'domain.pddl')


def get_station_agnostic_domain_filename():
    return osp.join(osp.dirname(__file__), 'domain-station-agnostic.pddl')
