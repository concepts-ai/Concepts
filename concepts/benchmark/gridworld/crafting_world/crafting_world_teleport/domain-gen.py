#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : domain-gen.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/06/2023
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

import concepts.benchmark.gridworld.crafting_world.crafting_world_rules as rules


def underline_to_pascal(s):
    return ''.join([w.capitalize() for w in s.split('_')])


mining_template_0 = """
 (:action {action_name}
  :parameters (?targetinv - inventory ?x - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?x ?t)
    (object-of-type ?x {target_type})
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target {create_type})
  )
 )
 (:regression {action_name}-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {create_type}))))
  :precondition (and (object-at ?target-resource ?t) (object-of-type ?target-resource {target_type}) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical))
  :rule (then
    (achieve (agent-at ?t))
    ({action_name} ?target-inventory ?target-resource ?target ?t)
  )
 )"""


mining_template_1 = """
 (:action {action_name}
  :parameters (?toolinv - inventory ?targetinv - inventory ?x - object ?tool - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?x ?t)
    (object-of-type ?x {target_type})
    (inventory-holding ?toolinv ?tool)
    (object-of-type ?tool {holding})
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target {create_type})
  )
 )
 (:regression {action_name}-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?holding - object) (forall ?holding-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {create_type}))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource {target_type}) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?holding-inventory ?holding) (object-of-type ?holding {holding})
  )
  :rule (then
    (achieve (agent-at ?t))
    ({action_name} ?holding-inventory ?target-inventory ?target-resource ?holding ?target ?t)
  )
 )
 (:regression {action_name}-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {create_type}))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {holding})))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {create_type})))))
  )
 )
 """

crafting_template_1 = """
 (:action {action_name}
  :parameters (?ingredientinv1 - inventory ?targetinv - inventory ?station - object ?ingredient1 - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?station ?t)
    (object-of-type ?station {station_type})
    (inventory-holding ?ingredientinv1 ?ingredient1)
    (object-of-type ?ingredient1 {ingredient1_type})
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target {create_type})
    (not (inventory-holding ?ingredientinv1 ?ingredient1))
    (inventory-empty ?ingredientinv1)
    (not (object-of-type ?ingredient1 {ingredient1_type}))
    (object-of-type ?ingredient1 Hypothetical)
  )
 )
 (:regression {action_name}-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?ingredient1 - object) (forall ?ingredient1-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {create_type}))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource {station_type}) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?ingredient1-inventory ?ingredient1) (object-of-type ?ingredient1 {ingredient1_type})
  )
  :rule (then
    (achieve (agent-at ?t))
    ({action_name} ?ingredient1-inventory ?target-inventory ?target-resource ?ingredient1 ?target ?t)
  )
 )
 (:regression {action_name}-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {create_type}))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {ingredient1_type})))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {create_type})))))
  )
 )
 """

crafting_template_2 = """
 (:action {action_name}
  :parameters (?ingredientinv1 - inventory ?ingredientinv2 - inventory ?targetinv - inventory ?station - object ?ingredient1 - object ?ingredient2 - object ?target - object ?t - tile)
  :precondition (and
    (agent-at ?t)
    (object-at ?station ?t)
    (object-of-type ?station {station_type})
    (inventory-holding ?ingredientinv1 ?ingredient1)
    (object-of-type ?ingredient1 {ingredient1_type})
    (inventory-holding ?ingredientinv2 ?ingredient2)
    (object-of-type ?ingredient2 {ingredient2_type})
    (inventory-empty ?targetinv)
    (object-of-type ?target Hypothetical)
  )
  :effect (and
    (not (inventory-empty ?targetinv))
    (inventory-holding ?targetinv ?target)
    (not (object-of-type ?target Hypothetical))
    (object-of-type ?target {create_type})
    (not (inventory-holding ?ingredientinv1 ?ingredient1))
    (inventory-empty ?ingredientinv1)
    (not (object-of-type ?ingredient1 {ingredient1_type}))
    (object-of-type ?ingredient1 Hypothetical)
    (not (inventory-holding ?ingredientinv2 ?ingredient2))
    (inventory-empty ?ingredientinv2)
    (not (object-of-type ?ingredient2 {ingredient2_type}))
    (object-of-type ?ingredient2 Hypothetical)
  )
 )
 (:regression {action_name}-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?ingredient1 - object) (forall ?ingredient1-inventory - inventory) (forall ?ingredient2 - object) (forall ?ingredient2-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {create_type}))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource {station_type}) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?ingredient1-inventory ?ingredient1) (object-of-type ?ingredient1 {ingredient1_type})
    (inventory-holding ?ingredient2-inventory ?ingredient2) (object-of-type ?ingredient2 {ingredient2_type})
  )
  :rule (then
    (achieve (agent-at ?t))
    ({action_name} ?ingredient1-inventory ?ingredient2-inventory ?target-inventory ?target-resource ?ingredient1 ?ingredient2 ?target ?t)
  )
 )
 (:regression {action_name}-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {create_type}))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {ingredient1_type})))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {ingredient2_type})))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {create_type})))))
  )
 )
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

    with open('./domain.pddl-template') as f:
        template = f.read()
    with open('./domain.pddl', 'w') as f:
        f.write(template.format(mining_rules=mining_rules, crafting_rules=crafting_rules))
    print('Generated: domain.pddl')


def main_station_agnostic():
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

    with open('./domain.pddl-template') as f:
        template = f.read()
    with open('./domain-station-agnostic.pddl', 'w') as f:
        f.write(template.format(mining_rules=mining_rules, crafting_rules=crafting_rules))
    print('Generated: domain.pddl')



if __name__ == '__main__':
    main()
    main_station_agnostic()

