#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : 2-simple-float-csp.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/10/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import concepts.dm.crow as crow

@crow.config_function_implementation
def f(x):
    return x ** 3


@crow.config_function_implementation
def inv_f(y):
    return y ** (1 / 3)


@crow.config_function_implementation(is_iterator=True)
def inv_plus(z, y):
    yield z - y


@crow.config_function_implementation(is_iterator=True)
def gen_float():
    for i in range(10):
        yield float(i)

