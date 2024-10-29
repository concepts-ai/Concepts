#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : crafting_world_pdsinterface.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/05/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Dict, Sequence

from concepts.dm.pdsketch.domain import Domain
from concepts.dm.pdsketch.operator import OperatorApplier
from concepts.benchmark.gridworld.crafting_world.crafting_world_env import CraftingWorldSimulator
from concepts.dm.pdsketch.strips.strips_expression import SStateDict

__all__ = ['CraftingWorldPDDLExecutor', 'reset_simulator_from_state']


class CraftingWorldPDDLExecutor(object):
    def __init__(self, domain: Domain, simulator: CraftingWorldSimulator):
        self.domain = domain
        self.simulator = simulator

    def execute(self, plan: Sequence[OperatorApplier]):
        last_failed_operator = None
        for i, action in enumerate(plan):
            rv = self.step(action)
            if not rv:
                last_failed_operator = i
                break

    def step(self, action: OperatorApplier) -> bool:
        action_name = action.operator.name
        action_args = action.arguments
        if action_name == "move-right":
            self.simulator.move_right()
        elif action_name == "move-left":
            self.simulator.move_left()
        elif action_name == "move-to":
            self.simulator.move_to(int(action_args[1][1:]))
        elif action_name == "pick-up":
            try:
                self.simulator.pick_up(int(_find_string_start_with(action_args, "i", first=True)[1:]), _find_string_start_with(action_args, "o", first=True))
            except KeyError as e:
                print(f'  pick-up {action_args} failed. Reason: {e}')
                return False
        elif action_name == "place-down":
            self.simulator.place_down(int(_find_string_start_with(action_args, "i", first=True)[1:]))
        elif action_name.startswith('mine'):
            # Trying mining.
            inventory_indices = [int(x[1:]) for x in _find_string_start_with(action_args, "i")]
            object_indices = _find_string_start_with(action_args, "o")

            hypothetical_object = [x for x in object_indices if x in self.simulator.hypothetical]
            if len(hypothetical_object) != 1:
                return False
            hypothetical_object = hypothetical_object[0]

            empty_inventory = [x for x in inventory_indices if self.simulator.inventory[x] is None]
            if len(empty_inventory) != 1:
                return False
            empty_inventory = empty_inventory[0]

            target_object = [
                x for x in object_indices if x in self.simulator.objects and self.simulator.objects[x][1] == self.simulator.agent_pos
            ]
            if len(target_object) != 1:
                return False
            target_object = target_object[0]

            tool_inventory = list(set(inventory_indices) - set([empty_inventory]))

            rv = self.simulator.mine(target_object, empty_inventory, hypothetical_object, tool_inventory=tool_inventory[0] if len(tool_inventory) > 0 else None)
            if not rv:
                return False
        elif action_name.startswith('craft'):
            inventory_indices = [int(x[1:]) for x in _find_string_start_with(action_args, "i")]
            object_indices = _find_string_start_with(action_args, "o")

            hypothetical_object = [x for x in object_indices if x in self.simulator.hypothetical]
            if len(hypothetical_object) != 1:
                return False
            hypothetical_object = hypothetical_object[0]

            empty_inventory = [x for x in inventory_indices if self.simulator.inventory[x] is None]
            if len(empty_inventory) != 1:
                return False
            empty_inventory = empty_inventory[0]

            target_object = [ x for x in object_indices if x in self.simulator.objects and self.simulator.objects[x][1] == self.simulator.agent_pos ]
            if len(target_object) != 1:
                return False
            target_object = target_object[0]

            ingredients = list(set(inventory_indices) - {empty_inventory})

            target_type = None
            hypothetical_object_index = action_args.index(hypothetical_object)
            hypothetical_object_varname = self.domain.operators[action_name].arguments[hypothetical_object_index].name
            for effect in self.domain.operators[action_name].effects:
                if (
                    effect.assign_expr.predicate.function.name == 'object-of-type' and
                    effect.assign_expr.predicate.arguments[0].name == hypothetical_object_varname and
                    effect.assign_expr.predicate.arguments[1].__class__.__name__ == 'ObjectConstantExpression' and
                    effect.assign_expr.value.__class__.__name__ == 'ConstantExpression' and
                    effect.assign_expr.value.constant.item() == 1
                ):
                    # print('  Found target type', effect.assign_expr)
                    target_type = effect.assign_expr.predicate.arguments[1].name
                    break

            rv = self.simulator.craft(target_object, empty_inventory, hypothetical_object, ingredients_inventory=ingredients, target_type=target_type)

            if not rv:
                return False
        else:
            return False

        return True


def reset_simulator_from_state(simulator: CraftingWorldSimulator, objects: Dict[str, Sequence[str]], state: SStateDict):
    agent_at = list(state['agent-at'])[0][0]
    simulator.agent_pos = int(agent_at[1:])

    simulator.nr_grids = len(objects['tile'])
    simulator.nr_inventory = nr_inventory = len(objects['inventory'])

    simulator.objects = dict()
    simulator.inventory = {i: None for i in range(1, 1 + nr_inventory)}

    for obj_name, obj_loc in state.get('object-at', []):
        obj_type = None
        for obj_name2, obj_type2 in state['object-of-type']:
            if obj_name2 == obj_name:
                obj_type = obj_type2
                break
        assert obj_type is not None
        simulator.objects[obj_name] = (obj_type, int(obj_loc[1:]))

    for inv_id, obj_name in state.get('inventory-holding', []):
        obj_type = None
        for obj_name2, obj_type2 in state['object-of-type']:
            if obj_name2 == obj_name:
                obj_type = obj_type2
        assert obj_type is not None
        simulator.inventory[int(inv_id[1:])] = (obj_type, obj_name)

    for obj_name, obj_type in state['object-of-type']:
        if obj_type == 'Hypothetical':
            simulator.hypothetical.add(obj_name)


def _find_string_start_with(list_of_string, start, first=False):
    rv = list()
    for s in list_of_string:
        if s.startswith(start):
            if first:
                return s
            rv.append(s)
    return rv
