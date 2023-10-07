#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : crafting_world_env.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/23/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
import os.path as osp
from PIL import Image
from typing import Sequence

from concepts.benchmark.gridworld.crafting_world.crafting_world_rules import MINING_RULES, CRAFTING_RULES, get_all_locations
from concepts.benchmark.gridworld.crafting_world.crafting_world_gen.utils import underline_to_pascal
from concepts.pdsketch.domain import Domain
from concepts.pdsketch.operator import OperatorApplier

SKIP_CRAFTING_LOCATION_CHECK = True


class CraftingWorldSimulator(object):
    def __init__(self):
        self.nr_grids = 15
        self.nr_inventory = 3
        self.agent_pos = 1
        self.objects = dict()  # str: (str, int), name: (type, pos)
        self.inventory = dict()  # int: Optional[Tuple[str, str]]  # (type, name)
        self.hypothetical = set()  # str

    def reset_from_state(self, objects, state):
        agent_at = list(state['agent-at'])[0][0]
        self.agent_pos = int(agent_at[1:])

        self.nr_grids = len(objects['tile'])
        self.nr_inventory = nr_inventory = len(objects['inventory'])

        self.objects = dict()
        self.inventory = {i: None for i in range(1, 1 + nr_inventory)}

        for obj_name, obj_loc in state.get('object-at', []):
            obj_type = None
            for obj_name2, obj_type2 in state['object-of-type']:
                if obj_name2 == obj_name:
                    obj_type = obj_type2
                    break
            assert obj_type is not None
            self.objects[obj_name] = (obj_type, int(obj_loc[1:]))

        for inv_id, obj_name in state.get('inventory-holding', []):
            obj_type = None
            for obj_name2, obj_type2 in state['object-of-type']:
                if obj_name2 == obj_name:
                    obj_type = obj_type2
            assert obj_type is not None
            self.inventory[int(inv_id[1:])] = (obj_type, obj_name)

        for obj_name, obj_type in state['object-of-type']:
            if obj_type == 'Hypothetical':
                self.hypothetical.add(obj_name)

    def move_to(self, pos):
        self.agent_pos = max(1, min(self.nr_grids, pos))
        return True

    def move_left(self):
        self.agent_pos = max(1, self.agent_pos - 1)
        return True

    def move_right(self):
        self.agent_pos = min(self.nr_grids, self.agent_pos + 1)
        return True

    def pick_up(self, inventory, obj_name):
        if self.inventory[inventory] is not None:
            return False
        if self.objects[obj_name][1] != self.agent_pos:
            return False

        self.inventory[inventory] = self.objects[obj_name][0], obj_name
        del self.objects[obj_name]
        return True

    def place_down(self, inventory):
        if self.inventory[inventory] is None:
            return False

        obj_type, obj_name = self.inventory[inventory]
        self.objects[obj_name] = obj_type, self.agent_pos

    def mine(self, obj_name, inventory, hypothetical_object_name, tool_inventory=None):
        if self.objects[obj_name][1] != self.agent_pos:
            return False
        if self.inventory[inventory] is not None:
            return False
        if hypothetical_object_name not in self.hypothetical:
            return False
        if tool_inventory is not None and self.inventory[tool_inventory] is None:
            return False

        obj_type, _ = self.objects[obj_name]

        for rule in MINING_RULES:
            if underline_to_pascal(rule['location']) == obj_type:
                if tool_inventory is None:
                    if len(rule['holding']) == 0:
                        new_obj_type = underline_to_pascal(rule['create'])
                        self.inventory[inventory] = (new_obj_type, hypothetical_object_name)
                        self.hypothetical.remove(hypothetical_object_name)
                        return True
                else:
                    tool_type, _ = self.inventory[tool_inventory]
                    if len(rule['holding']) == 0 or (len(rule['holding']) == 1 and underline_to_pascal(rule['holding'][0]) == tool_type):
                        new_obj_type = underline_to_pascal(rule['create'])
                        self.inventory[inventory] = (new_obj_type, hypothetical_object_name)
                        self.hypothetical.remove(hypothetical_object_name)
                        return True

        return False

    def craft(self, obj_name, inventory, hypothetical_object_name, ingredients_inventory, target_type=None):
        if SKIP_CRAFTING_LOCATION_CHECK:
            if self.objects[obj_name][1] != self.agent_pos:
                return False
        if self.inventory[inventory] is not None:
            return False
        if hypothetical_object_name not in self.hypothetical:
            return False
        for ingredient_inventory in ingredients_inventory:
            if self.inventory[ingredient_inventory] is None:
                return False

        obj_type, _ = self.objects[obj_name]
        # print('Checking crafting', inventory, hypothetical_object_name, target_type, ingredients_inventory)

        for rule in CRAFTING_RULES:
            if target_type is not None and underline_to_pascal(rule['create']) != target_type:
                continue
            # print('  checking crafting rule', rule['location'], rule['recipe'], rule['create'])
            # if not SKIP_CRAFTING_LOCATION_CHECK:
            #     print(f'    matching crafting location', underline_to_pascal(rule['location']), obj_type)
            if underline_to_pascal(rule['location']) == obj_type or SKIP_CRAFTING_LOCATION_CHECK:
                if len(rule['recipe']) == len(ingredients_inventory):
                    current_holding_types = set()
                    for ingredient_inventory in ingredients_inventory:
                        ingredient_type, _ = self.inventory[ingredient_inventory]
                        current_holding_types.add(ingredient_type)
                    target_holding_types = set()
                    for ingredient_type in rule['recipe']:
                        target_holding_types.add(underline_to_pascal(ingredient_type))
                    # print(f'    matching crafting recipe current={current_holding_types}, target={target_holding_types}')
                    if current_holding_types == target_holding_types:
                        new_obj_type = underline_to_pascal(rule['create'])
                        self.inventory[inventory] = (new_obj_type, hypothetical_object_name)
                        self.hypothetical.remove(hypothetical_object_name)
                        for ingredient_inventory in ingredients_inventory:
                            self.hypothetical.add(self.inventory[ingredient_inventory][1])
                            self.inventory[ingredient_inventory] = None
                        return True
        return False


class CraftingWorldRenderer(object):
    def __init__(self, map_w: int, map_h: int, max_inventory: int):
        self._map_w = map_w
        self._map_h = map_h
        self.basic_canvas = np.zeros((map_h * 17 + 1, map_w * 17 + 1, 3), dtype=np.uint8) + 255
        self._inventory_h = int(max_inventory / map_w) + (1 if max_inventory % map_w != 0 else 0)
        self.inventory_canvas = np.zeros((self._inventory_h * 17 + 1, map_w * 17 + 1, 3), dtype=np.uint8) + 255
        self._init_basic_canvas()
        self._block_images = dict()
        self._init_resource_png()

    def _init_basic_canvas(self):
        # Draw grid
        for i in range(self._map_w + 1):
            self.basic_canvas[:, i * 17, :] = 128
        for i in range(self._map_h + 1):
            self.basic_canvas[i * 17, :, :] = 128

        # Draw grid for the inventory
        for i in range(self._inventory_h + 1):
            self.inventory_canvas[i * 17, :, :] = 128
        for i in range(self._map_w + 1):
            self.inventory_canvas[:, i * 17, :] = 128

    def _init_resource_png(self):
        filename = osp.join(osp.dirname(__file__), 'assets', 'BlockCSS.png')
        block_image = np.array(Image.open(filename))
        filename = osp.join(osp.dirname(__file__), 'assets', 'ItemCSS.png')
        item_image = np.array(Image.open(filename))
        filename = osp.join(osp.dirname(__file__), 'assets', 'EntityCSS.png')
        entity_image = np.array(Image.open(filename))

        def extract(image, x, y):
            return image[y:y+16, x:x+16, :]

        def make_small_item_image(image):
            item_image = np.zeros_like(image)
            # item_image[8:8+8, :8, :] = image[::2, ::2, :]
            item_image[8:8+8, :8, :3] = image[::2, ::2, :3] * (image[::2, ::2, 3:4] / 255) + 255 * (1 - image[::2, ::2, 3:4] / 255)
            item_image[8:8+8, :8, 3] = 255
            # add a small border
            item_image[8:8+8, 0, :] = 128
            item_image[8:8+8, 7, :] = 128
            item_image[8, :8, :] = 128
            item_image[15, :8, :] = 128
            return item_image

        def make_small_villager_image(image):
            villager_image = np.zeros((image.shape[0], image.shape[1], 4))
            villager_image[8:8+8, 8:8+8, :3] = image[::2, ::2, :]
            villager_image[8:8+8, 8:8+8, 3] = (image[::2, ::2].mean(axis=-1) < 255-32) * 255
            return villager_image

        self._block_images['GoldOreVein'] = extract(block_image, 752, 448)
        self._block_images['CoalOreVein'] = extract(block_image, 704, 448)
        self._block_images['IronOreVein'] = extract(block_image, 0, 464)
        self._block_images['CobblestoneStash'] = extract(block_image, 32, 416)
        self._block_images['BeetrootCrop'] = extract(item_image, 96, 480)
        self._block_images['Chicken'] = extract(entity_image, 32, 432)
        self._block_images['Sheep'] = extract(entity_image, 144, 432)
        self._block_images['Tree'] = np.array(Image.open(osp.join(osp.dirname(__file__), 'assets', 'Jungle_Tree.png')).resize((16, 16)))
        self._block_images['SugarCanePlant'] = extract(block_image, 432, 576)
        self._block_images['PotatoPlant'] = extract(block_image, 304, 576)
        self._block_images['WorkStation'] = extract(block_image, 0, 512)
        self._block_images['Furnace'] = extract(block_image, 464, 464)

        self._block_images['Coal'] = extract(item_image, 176, 544)
        self._block_images['Coal/Small'] = make_small_item_image(self._block_images['Coal'])
        self._block_images['Feather'] = extract(item_image, 240, 544)
        self._block_images['Feather/Small'] = make_small_item_image(self._block_images['Feather'])
        self._block_images['Axe'] = extract(item_image, 160, 464)  # Stone Axe
        self._block_images['Axe/Small'] = make_small_item_image(self._block_images['Axe'])
        self._block_images['Pickaxe'] = extract(item_image, 192, 464)  # Stone Pickaxe
        self._block_images['Pickaxe/Small'] = make_small_item_image(self._block_images['Pickaxe'])
        self._block_images['SugarCane'] = extract(item_image, 176, 576)
        self._block_images['SugarCane/Small'] = make_small_item_image(self._block_images['SugarCane'])
        self._block_images['Potato'] = extract(item_image, 224, 496)
        self._block_images['Potato/Small'] = make_small_item_image(self._block_images['Potato'])
        self._block_images['CookedPotato'] = extract(item_image, 64, 480)
        self._block_images['CookedPotato/Small'] = make_small_item_image(self._block_images['CookedPotato'])
        self._block_images['Beetroot'] = extract(item_image, 96, 480)
        self._block_images['Beetroot/Small'] = make_small_item_image(self._block_images['Beetroot'])
        self._block_images['BeetrootSoup'] = extract(item_image, 112, 480)
        self._block_images['BeetrootSoup/Small'] = make_small_item_image(self._block_images['BeetrootSoup'])
        self._block_images['Bed'] = extract(block_image, 208, 528)
        self._block_images['Bed/Small'] = make_small_item_image(self._block_images['Bed'])
        self._block_images['IronOre'] = extract(item_image, 0, 880)
        self._block_images['IronOre/Small'] = make_small_item_image(self._block_images['IronOre'])
        self._block_images['IronIngot'] = extract(item_image, 112, 560)
        self._block_images['IronIngot/Small'] = make_small_item_image(self._block_images['IronIngot'])
        self._block_images['GoldOre'] = extract(item_image, 240, 864)
        self._block_images['GoldOre/Small'] = make_small_item_image(self._block_images['GoldOre'])
        self._block_images['GoldIngot'] = extract(item_image, 48, 560)
        self._block_images['GoldIngot/Small'] = make_small_item_image(self._block_images['GoldIngot'])
        self._block_images['Cobblestone'] = extract(item_image, 240, 800)  # actually Netherite Ingot
        self._block_images['Cobblestone/Small'] = make_small_item_image(self._block_images['Cobblestone'])
        self._block_images['Sword'] = extract(item_image, 0, 448)  # Iron Sword
        self._block_images['Sword/Small'] = make_small_item_image(self._block_images['Sword'])
        self._block_images['Shears'] = extract(item_image, 144, 464)
        self._block_images['Shears/Small'] = make_small_item_image(self._block_images['Shears'])
        self._block_images['Stick'] = extract(item_image, 80, 48)
        self._block_images['Stick/Small'] = make_small_item_image(self._block_images['Stick'])
        self._block_images['Boat'] = extract(item_image, 112, 592)  # Oak Boat
        self._block_images['Boat/Small'] = make_small_item_image(self._block_images['Boat'])
        self._block_images['Bowl'] = extract(item_image, 208, 528)
        self._block_images['Bowl/Small'] = make_small_item_image(self._block_images['Bowl'])
        self._block_images['Wood'] = extract(block_image, 352, 416)  # Oak Log
        self._block_images['Wood/Small'] = make_small_item_image(self._block_images['Wood'])
        self._block_images['WoodPlank'] = extract(block_image, 224, 464)  # Oak Planks
        self._block_images['WoodPlank/Small'] = make_small_item_image(self._block_images['WoodPlank'])
        self._block_images['Wool'] = extract(block_image, 0, 32)  # White Wool
        self._block_images['Wool/Small'] = make_small_item_image(self._block_images['Wool'])
        self._block_images['Arrow'] = extract(item_image, 80, 32)
        self._block_images['Arrow/Small'] = make_small_item_image(self._block_images['Arrow'])
        self._block_images['Paper'] = extract(item_image, 96, 544)
        self._block_images['Paper/Small'] = make_small_item_image(self._block_images['Paper'])

        self._block_images['agent'] = np.array(Image.open(osp.join(osp.dirname(__file__), 'assets', 'Plains_Villager_Base.png')).resize((16, 16)))
        self._block_images['agent/Small'] = make_small_villager_image(self._block_images['agent'])

    def render(self, simulator: CraftingWorldSimulator):
        map_canvas = self.basic_canvas.copy()
        inventory_canvas = self.inventory_canvas.copy()

        def draw_primitive(canvas, obj_pos, image, mask=None):
            obj_x = obj_pos % self._map_w
            obj_y = int(obj_pos / self._map_w)
            if isinstance(image, np.ndarray) and image.shape[2] == 4:
                image, mask = image[:, :, :3], image[:, :, 3]
            if mask is not None:
                canvas[obj_y * 17 + 1: (obj_y + 1) * 17, obj_x * 17 + 1: (obj_x + 1) * 17, :] = np.clip((
                    image * (mask[:, :, None] / 255) +
                    canvas[obj_y * 17 + 1: (obj_y + 1) * 17, obj_x * 17 + 1: (obj_x + 1) * 17, :] * (1 - mask[:, :, None] / 255)
                ), 0, 255).astype('uint8')
            else:
                canvas[obj_y * 17 + 1: (obj_y + 1) * 17, obj_x * 17 + 1: (obj_x + 1) * 17, :] = image

        def draw(canvas, obj_name, obj_type, obj_pos, use_small=True):
            if obj_pos > self._map_w * self._map_h:
                raise ValueError(f'Object {obj_name} is out of bounds')
            obj_pos = obj_pos - 1

            if obj_type.endswith('Station'):
                obj_type = 'WorkStation'

            if use_small and obj_type + '/Small' in self._block_images:
                image = self._block_images[obj_type + '/Small']
            elif obj_type in self._block_images:
                image = self._block_images[obj_type]
            else:
                print(f'Warning: unknown object type {obj_type}.')
                image = 0

            draw_primitive(canvas, obj_pos, image)

        # Draw objects
        location_types = get_all_locations()
        for obj_name, (obj_type, obj_pos) in simulator.objects.items():
            if obj_type in location_types:
                draw(map_canvas, obj_name, obj_type, obj_pos)
        for obj_name, (obj_type, obj_pos) in simulator.objects.items():
            if obj_type not in location_types:
                draw(map_canvas, obj_name, obj_type, obj_pos)
        for i, content in simulator.inventory.items():
            if content is not None:
                obj_type, obj_name = content
                draw(inventory_canvas, obj_name, obj_type, i, use_small=False)

        # Highlight the border of the agent position
        agent_x = (simulator.agent_pos - 1) % self._map_w
        agent_y = int((simulator.agent_pos - 1) / self._map_w)
        map_canvas[agent_y * 17: (agent_y + 1) * 17 + 1, agent_x * 17, :] = (29, 209, 77)
        map_canvas[agent_y * 17: (agent_y + 1) * 17 + 1, (agent_x + 1) * 17, :] = (29, 209, 77)
        map_canvas[agent_y * 17, agent_x * 17: (agent_x + 1) * 17 + 1, :] = (29, 209, 77)
        map_canvas[(agent_y + 1) * 17, agent_x * 17: (agent_x + 1) * 17 + 1, :] = (29, 209, 77)
        draw_primitive(map_canvas, simulator.agent_pos - 1, self._block_images['agent/Small'])

        # Concatenate the two canvases
        canvas = np.concatenate((map_canvas, np.zeros((8, map_canvas.shape[1], 3), dtype='uint8') + 255, inventory_canvas), axis=0)
        return canvas


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


def _find_string_start_with(list_of_string, start, first=False):
    rv = list()
    for s in list_of_string:
        if s.startswith(start):
            if first:
                return s
            rv.append(s)
    return rv
