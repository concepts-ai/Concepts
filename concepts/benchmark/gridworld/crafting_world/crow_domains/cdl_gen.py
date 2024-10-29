#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cdl_gen.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 06/21/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp
import concepts.benchmark.gridworld.crafting_world.crafting_world_rules as rules

g_this_dir = osp.dirname(__file__)


def underline_to_pascal(s):
    return ''.join([w.capitalize() for w in s.split('_')])


mining_template_0 = """
behavior {action_name}():
  goal:
    exists i: Inventory where: (
      exists x: Object where: (
        inventory_holding(i, x) and object_of_type(x, {create_type})
      )
    )
  body:
    bind pos: Tile, r: Object where:
      object_at(r, pos) and object_of_type(r, {target_type})
    bind i: Inventory, x: Object where:
      inventory_empty(i) and object_of_type(x, Hypothetical)
    achieve agent_at(pos)
    do ctl_mine_0(i, x, r, pos, {target_type})
  eff:
    inventory_empty[i] = False
    inventory_holding[i, x] = True
    object_of_type[x, {create_type}] = True
    object_of_type[x, Hypothetical] = False
"""

mining_template_1 = """
behavior {action_name}():
  goal: 
    exists i: Inventory where: (
      exists x: Object where: (
        inventory_holding(i, x) and object_of_type(x, {create_type})
      )
    )
  body:
    achieve exists _i: Inventory where: (
      exists _x: Object where: (
        inventory_holding(_i, _x) and object_of_type(_x, {holding})
      )
    )
    bind ti: Inventory, t: Object where:
      inventory_holding(ti, t) and object_of_type(t, {holding})
    bind pos: Tile, r: Object where:
      object_at(r, pos) and object_of_type(r, {target_type})
    bind i: Inventory, x: Object where:
      inventory_empty(i) and object_of_type(x, Hypothetical)
    achieve agent_at(pos)
    do ctl_mine_1(i, x, ti, t, r, pos, {target_type})
  eff:
    inventory_empty[i] = False
    inventory_holding[i, x] = True
    object_of_type[x, {create_type}] = True
    object_of_type[x, Hypothetical] = False
"""

crafting_template_1 = """
behavior {action_name}():
  goal:
    exists i: Inventory where: (
      exists x: Object where: (
        inventory_holding(i, x) and object_of_type(x, {create_type})
      )
    )
  body:
    achieve exists _i: Inventory where: (
      exists _x: Object where: (
        inventory_holding(_i, _x) and object_of_type(_x, {ingredient1_type})
      )
    )
    bind yi: Inventory, y: Object where:
      inventory_holding(yi, y) and object_of_type(y, {ingredient1_type})
    bind pos: Tile, s: Object where:
      object_at(s, pos) and object_of_type(s, {station_type})
    bind i: Inventory, x: Object where:
      inventory_empty(i) and object_of_type(x, Hypothetical)
    achieve agent_at(pos)
    do ctl_craft_1(i, x, yi, y, s, pos, {create_type})
  eff:
    inventory_empty[i] = False
    inventory_holding[i, x] = True
    object_of_type[x, {create_type}] = True
    object_of_type[x, Hypothetical] = False
    inventory_empty[yi] = True
    inventory_holding[yi, y] = False
    object_of_type[y, {ingredient1_type}] = False
    object_of_type[y, Hypothetical] = True
"""


crafting_template_2 = """
behavior {action_name}():
  goal:
    exists i: Inventory where: (
      exists x: Object where: (
        inventory_holding(i, x) and object_of_type(x, {create_type})
      )
    )
  body:
    achieve exists _i: Inventory where: (
      exists _x: Object where: (
        inventory_holding(_i, _x) and object_of_type(_x, {ingredient1_type})
      )
    )
    achieve exists _i: Inventory where: (
      exists _x: Object where: (
        inventory_holding(_i, _x) and object_of_type(_x, {ingredient2_type})
      )
    )
    bind yi: Inventory, y: Object where:
      inventory_holding(yi, y) and object_of_type(y, {ingredient1_type})
    bind zi: Inventory, z: Object where:
      inventory_holding(zi, z) and object_of_type(z, {ingredient2_type})
    bind pos: Tile, s: Object where:
      object_at(s, pos) and object_of_type(s, {station_type})
    bind i: Inventory, x: Object where:
     inventory_empty(i) and object_of_type(x, Hypothetical)
    achieve agent_at(pos)
    do ctl_craft_2(i, x, yi, y, zi, z, s, pos, {create_type})
  eff:
    inventory_empty[i] = False
    inventory_holding[i, x] = True
    object_of_type[x, {create_type}] = True
    object_of_type[x, Hypothetical] = False
    inventory_empty[yi] = True
    inventory_holding[yi, y] = False
    object_of_type[y, {ingredient1_type}] = False
    object_of_type[y, Hypothetical] = True
    inventory_empty[zi] = True
    inventory_holding[zi, z] = False
    object_of_type[z, {ingredient2_type}] = False
    object_of_type[z, Hypothetical] = True
"""


def main():
    mining_rules = ''
    for r in rules.MINING_RULES:
        action_name = r['rule_name'].replace('_', '-')
        create_type = underline_to_pascal(r['create'])
        target_type = underline_to_pascal(r['location'])
        holding = r['holding'][0] if len(r['holding']) == 1 else None
        if holding is not None:
            holding = underline_to_pascal(holding)

        if holding is None:
            mining_rules += mining_template_0.format(action_name=action_name, create_type=create_type, target_type=target_type)
        else:
            mining_rules += mining_template_1.format(action_name=action_name, create_type=create_type, target_type=target_type, holding=holding)

    crafting_rules = ''
    for r in rules.CRAFTING_RULES:
        action_name = r['rule_name'].replace('_', '-')
        create_type = underline_to_pascal(r['create'])
        station_type = underline_to_pascal(r['location'])
        recipe = list(map(underline_to_pascal, r['recipe']))

        if len(recipe) == 1:
            ingredient1_type = recipe[0]
            crafting_rules += crafting_template_1.format(action_name=action_name, create_type=create_type, station_type=station_type, ingredient1_type=ingredient1_type)
        elif len(recipe) == 2:
            ingredient1_type, ingredient2_type = recipe
            crafting_rules += crafting_template_2.format(action_name=action_name, create_type=create_type, station_type=station_type, ingredient1_type=ingredient1_type, ingredient2_type=ingredient2_type)
        else:
            raise ValueError('Invalid recipe length: {}'.format(len(recipe)))

    with open(osp.join(g_this_dir, 'domain.pddl-template')) as f:
        template = f.read()
    with open(osp.join(g_this_dir, 'domain.pddl'), 'w') as f:
        f.write(template.format(mining_rules=mining_rules, crafting_rules=crafting_rules))
    print('Generated: domain.pddl')


def main_station_agnostic():
    mining_rules = ''
    for r in rules.MINING_RULES:
        action_name = r['rule_name']
        create_type = underline_to_pascal(r['create'])
        target_type = underline_to_pascal(r['location'])
        holding = r['holding'][0] if len(r['holding']) == 1 else None
        if holding is not None:
            holding = underline_to_pascal(holding)

        if holding is None:
            mining_rules += mining_template_0.format(action_name=action_name, create_type=create_type, target_type=target_type)
        else:
            mining_rules += mining_template_1.format(action_name=action_name, create_type=create_type, target_type=target_type, holding=holding)

    crafting_rules = ''
    for r in rules.CRAFTING_RULES:
        action_name = r['rule_name']
        create_type = underline_to_pascal(r['create'])
        station_type = 'WorkStation'
        recipe = list(map(underline_to_pascal, r['recipe']))

        if len(recipe) == 1:
            ingredient1_type = recipe[0]
            crafting_rules += crafting_template_1.format(action_name=action_name, create_type=create_type, station_type=station_type, ingredient1_type=ingredient1_type)
        elif len(recipe) == 2:
            ingredient1_type, ingredient2_type = recipe
            crafting_rules += crafting_template_2.format(action_name=action_name, create_type=create_type, station_type=station_type, ingredient1_type=ingredient1_type, ingredient2_type=ingredient2_type)
        else:
            raise ValueError('Invalid recipe length: {}'.format(len(recipe)))

    with open(osp.join(g_this_dir, 'crafting_world.cdl-template')) as f:
        template = f.read()
    with open(osp.join(g_this_dir, 'crafting_world_station_agnostic.cdl'), 'w') as f:
        f.write(template.format(mining_rules=mining_rules, crafting_rules=crafting_rules))
    print('Generated: crafting_world_station_agnostic.cdl')



if __name__ == '__main__':
    # main()
    main_station_agnostic()
