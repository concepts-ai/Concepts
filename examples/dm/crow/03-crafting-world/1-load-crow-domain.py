#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : 1-load-crow-domain.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/17/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import jacinle
import concepts.dm.crow as crow
from concepts.benchmark.gridworld.crafting_world.crow_domains import get_crafting_world_domain_filename


domain = crow.load_domain_file(get_crafting_world_domain_filename())


print(domain.name)
table = list()
for k, v in domain.types.items():
    table.append((f'Type::{k}', str(v)))
for k, v in domain.constants.items():
    table.append((f'Constant::{k}', str(v)))
for k, v in domain.features.items():
    table.append((f'Feature::{k}', str(v)))
for k, v in domain.functions.items():
    table.append((f'Function::{k}', str(v)))
for k, v in domain.controllers.items():
    table.append((f'Controller::{k}', str(v)))
for k, v in domain.behaviors.items():
    table.append((f'Behavior::{k}', str(v)))

print(jacinle.tabulate(table))

