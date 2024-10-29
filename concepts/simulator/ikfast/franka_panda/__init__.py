#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/24/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from jacinle.jit.cext import auto_travis

auto_travis(__file__, required_imports=['ikfast_panda_arm'])
