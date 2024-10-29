#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : 2-visualize-oracle-policy.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 06/24/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import jacinle
import concepts.dm.crow as crow
from concepts.simulator.pybullet.client import BulletClient
from concepts.benchmark.manip_tabletop.paint_factory.paint_factory import PaintFactoryEnv
from concepts.benchmark.manip_tabletop.paint_factory.paint_factory_policy import PaintFactoryOraclePolicy
from concepts.benchmark.manip_tabletop.paint_factory.crow_domains import get_paint_factory_domain_filename
from concepts.benchmark.manip_tabletop.paint_factory.crow_domains.paint_factory_crow_interface import PaintFactoryPerceptionInterface


domain_filename = get_paint_factory_domain_filename()
domain = crow.load_domain_file(domain_filename)

client = BulletClient(is_gui=True, render_fps=240)
env = PaintFactoryEnv(client)
env.reset()

percept = PaintFactoryPerceptionInterface(env, domain)
policy = PaintFactoryOraclePolicy(env)

state = percept.get_crow_state()
print(jacinle.colored('Initial State:', 'yellow'))
print(state)

object_images = state['object_image']
images = list()
images_titles = list()
for i, name in enumerate(state.object_type2name['Object']):
    image = object_images.tensor[i].reshape(32, 32, 3)
    image = image.numpy().astype('uint8')
    images.append(image)
    images_titles.append(name)

from jaclearn.visualize.imgrid import auto_image_grid_mplib

auto_image_grid_mplib(images, images_titles, global_title='object_image')

print(jacinle.colored('Oracle policy execution:', 'green'))
for i in range(100):  # A sufficient number of steps to reach the final state
    action_dict = policy.act()
    if action_dict is None:
        break
    print('  Executing:', action_dict)
    env.step_object_name(**action_dict)

print(jacinle.colored('Oracle policy execution finished.', 'green'))
print(jacinle.colored(f'Goal achieved? {env.is_goal_achieved()}', 'green'))

print(jacinle.colored('Final State:', 'yellow'))
state = percept.get_crow_state()
print(state)

print(jacinle.colored('Press Ctrl+C to exit.', 'yellow'))
client.wait_forever()
