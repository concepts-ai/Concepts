#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dirty-feature-implementations.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/08/2025
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import concepts.dm.crow as crow


@crow.config_function_implementation()
def is_okay(x):
     return False
