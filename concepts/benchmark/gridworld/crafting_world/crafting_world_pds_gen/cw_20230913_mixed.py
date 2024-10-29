#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cw_20230913_mixed.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 06/18/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
import numpy.random as npr
from typing import Any, Optional, Tuple, Dict

from concepts.benchmark.gridworld.crafting_world.crafting_world_rules import get_all_mining_locations, get_all_mining_outcomes
from concepts.benchmark.gridworld.crafting_world.crafting_world_rules import get_all_crafting_ingradients, get_all_crafting_locations, get_all_crafting_outcomes, CRAFTING_RULES
from concepts.benchmark.gridworld.crafting_world.utils import underline_to_pascal, underline_to_space


PROBLEM_PDDL_TEMPLATE = """
(define
 (problem crafting-world-v20230106-p{problem_id})
 (:domain crafting-world-v20230106)
 (:objects
   {objects_str}
 )
 (:init
   {init_str}
 )
 (:goal {goal_str} )
)
"""


def gen_linear_tile(n: int) -> Tuple[str, str]:
    """Generate the object string and init string for a linear 1D map.

    .. code::

        t1 t2 t3 t4 ... tn

    Args:
        n: the length of the map.

    Returns:
        a tuple of the object string and init string.
    """

    object_str = ""
    init_str = ""
    for i in range(1, n + 1):
        object_str += f"t{i} - tile\n"
        # if i > 1:
        #     init_str += f"(tile-right t{i - 1} t{i})\n"
        #     init_str += f"(tile-left t{i} t{i - 1})\n"
    return object_str, init_str


def gen_locations_and_objects(n: int, inventory_size: int, station_agnostic: bool = True) -> Tuple[str, str, int, list, list]:
    """Sample a linear map by randomly placing mining locations and tools on the map."""

    all_locations = get_all_mining_locations() + get_all_crafting_locations(use_only_workstation=station_agnostic)
    mining_tools = ['axe', 'pickaxe']  #  + ['coal'] * 5

    nr_objects = len(all_locations) + len(mining_tools) + inventory_size

    map_location = [None for _ in range(n)]
    map_objects = [[] for _ in range(n)]

    assert len(all_locations) < n, 'Map is too small to fit all locations: {} < {}'.format(n, len(all_locations))

    object_str = ''
    init_str = ''

    location_assignments = npr.choice(np.arange(n), len(all_locations), replace=False)
    for i, x in enumerate(location_assignments):
        map_location[x] = (i + 1, all_locations[i])
        object_str += f'o{i + 1} - object\n'
        init_str += '(object-of-type o{} {})\n'.format(i + 1, underline_to_pascal(all_locations[i]))
        init_str += '(object-at o{} t{})\n'.format(i + 1, x + 1)

    mining_tool_assignments = npr.choice(np.arange(n), len(mining_tools), replace=False)
    for i, x in enumerate(mining_tool_assignments):
        map_objects[x].append((i + 1 + len(all_locations), mining_tools[i]))
        object_str += f'o{i + 1 + len(all_locations)} - object\n'
        init_str += '(object-of-type o{} {})\n'.format(i + 1 + len(all_locations), underline_to_pascal(mining_tools[i]))
        init_str += '(object-at o{} t{})\n'.format(i + 1 + len(all_locations), x + 1)

    for i in range(inventory_size):
        object_str += f'o{i + 1 + len(all_locations) + len(mining_tools)} - object\n'
        object_str += f'i{i + 1} - inventory\n'
        init_str += '(object-of-type o{} Hypothetical)\n'.format(i + 1 + len(all_locations) + len(mining_tools))
        init_str += '(inventory-empty i{})\n'.format(i + 1)

    return object_str, init_str, nr_objects, map_location, map_objects


def gen_v20230913_instance_record(problem_id: str, split: str, n: int = 25, inventory_size: int = 8, goal_index: Optional[int] = None, station_agnostic: bool = True) -> Dict[str, Any]:
    import jacinle

    all_outcomes = get_all_crafting_outcomes() + get_all_mining_outcomes()
    if goal_index is not None:
        assert goal_index < len(all_outcomes), 'Invalid goal index: {} >= {}'.format(goal_index, len(all_outcomes))
        goal = all_outcomes[goal_index]
    else:
        goal = npr.choice(all_outcomes)

    obj_object_str, obj_init_str, nr_objects, map_location, map_objects = gen_locations_and_objects(n, inventory_size)
    map_object_str, map_init_str = gen_linear_tile(n)
    object_str = obj_object_str + map_object_str
    init_str = obj_init_str + map_init_str
    init_str += '(agent-at t1)\n'

    goal_nl = 'Mine or craft a {}.'.format(underline_to_space(goal))
    # goal_str = f'(and (inventory-holding i{inventory_size} o{nr_objects})\n(object-of-type o{nr_objects} {underline_to_pascal(goal)}))\n'
    goal_str = f'(exists (?i - inventory) (exists (?o - object) (and (inventory-holding ?i ?o) (object-of-type ?o {underline_to_pascal(goal)})) ))'

    problem_pddl = PROBLEM_PDDL_TEMPLATE.format(
        problem_id=problem_id,
        objects_str=jacinle.indent_text(object_str, indent_format='   ').strip(),
        init_str=jacinle.indent_text(init_str, indent_format='   ').strip(),
        goal_str=jacinle.indent_text(goal_str, indent_format='   ').strip(),
    )

    return dict(
        problem_id=problem_id,
        split=split,
        problem_pddl=problem_pddl,
        goal_nl=goal_nl,
        goal=goal,
        map_location=map_location,
        map_objects=map_objects,
        nr_objects=nr_objects,
        inventory_size=inventory_size,
    )
