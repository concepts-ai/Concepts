#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : 1-visualize-env.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/7/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import argparse
from concepts.benchmark.manip_tabletop.pick_place_hierarchy.pick_place_hierarchy import create_environment, get_available_tasks


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='h0-simple', choices=get_available_tasks())
args = parser.parse_args()

client, metainfo = create_environment(args.task)
client.wait_forever()
