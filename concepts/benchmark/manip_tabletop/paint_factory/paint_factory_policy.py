#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : paint_factory_policy.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/24/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from concepts.benchmark.manip_tabletop.paint_factory.paint_factory import PaintFactoryEnv


class PaintFactoryOraclePolicy(object):
    def __init__(self, env: PaintFactoryEnv):
        self._env = env

    def act(self, obs):
        return