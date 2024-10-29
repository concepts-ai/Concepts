#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : 2-simple-float-csp.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/10/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp
import jacinle
import concepts.dm.crow as crow

@crow.config_function_implementation
def f(x):
    return x ** 3


@crow.config_function_implementation
def inv_f(y):
    return y ** (1 / 3)


@crow.config_function_implementation(is_iterator=True)
def inv_plus(z):
    yield z / 2, z / 2


def main():
    problem = crow.load_problem_file(osp.join(osp.dirname(__file__), '2-simple-float-csp.cdl'))
    executor = problem.domain.make_executor()
    executor.register_function_implementation('f', f)
    executor.register_function_implementation('inv_f', inv_f)
    executor.register_function_implementation('inv_plus', inv_plus)
    plans, _ = crow.crow_regression(executor, problem)
    for x in plans[0]:
        print(x)


if __name__ == '__main__':
    main()
