from functools import lru_cache
from typing import List

MINING_RULES = [
    dict(
        rule_name='mine_iron_ore',
        create='iron_ore',
        action='mine',
        location='iron_ore_vein',
        recipe=[],
        holding=['pickaxe'],
    ),
    dict(
        rule_name='mine_coal',
        create='coal',
        action='mine',
        location='coal_ore_vein',
        recipe=[],
        holding=['pickaxe'],
    ),
    dict(
        rule_name='mine_cobblestone',
        create='cobblestone',
        action='mine',
        location='cobblestone_stash',
        recipe=[],
        holding=['pickaxe'],
    ),
    dict(
        rule_name='mine_wood',
        create='wood',
        action='mine',
        location='tree',
        recipe=[],
        holding=['axe'],
    ),
    dict(
        rule_name='mine_feather',
        create='feather',
        action='mine',
        location='chicken',
        recipe=[],
        holding=['sword'],
    ),
    dict(
        rule_name='mine_wool1',
        create='wool',
        action='mine',
        location='sheep',
        recipe=[],
        holding=['shears'],
    ),
    dict(
        rule_name='mine_wool2',
        create='wool',
        action='mine',
        location='sheep',
        recipe=[],
        holding=['sword'],
    ),
    dict(
        rule_name='mine_potato',
        create='potato',
        action='mine',
        location='potato_plant',
        recipe=[],
        holding=[],
    ),
    dict(
        rule_name='mine_beetroot',
        create='beetroot',
        action='mine',
        location='beetroot_crop',
        recipe=[],
        holding=[],
    ),
    dict(
        rule_name='mine_gold_ore',
        create='gold_ore',
        action='mine',
        location='gold_ore_vein',
        recipe=[],
        holding=['pickaxe'],
    ),
    dict(
        rule_name='mine_sugar_cane',
        create='sugar_cane',
        action='mine',
        location='sugar_cane_plant',
        recipe=[],
        holding=[],
    ),
]

CRAFTING_RULES = [
    dict(
        rule_name='craft_wood_plank',
        create='wood_plank',
        action='craft',
        location='work_station',
        recipe=['wood'],
        holding=[],
    ),
    dict(
        rule_name='craft_stick',
        create='stick',
        action='craft',
        location='work_station',
        recipe=['wood_plank'],
        holding=[],
    ),
    dict(
        rule_name='craft_arrow',
        create='arrow',
        action='craft',
        location='weapon_station',
        recipe=['feather', 'stick'],
        holding=[],
    ),
    dict(
        rule_name='craft_sword',
        create='sword',
        action='craft',
        location='weapon_station',
        recipe=['iron_ingot', 'stick'],
        holding=[],
    ),
    dict(
        rule_name='craft_shears1',
        create='shears',
        action='craft',
        location='tool_station',
        recipe=['iron_ingot'],
        holding=[],
    ),
    dict(
        rule_name='craft_shears2',
        create='shears',
        action='craft',
        location='tool_station',
        recipe=['gold_ingot'],
        holding=[],
    ),
    dict(
        rule_name='craft_iron_ingot',
        create='iron_ingot',
        action='craft',
        location='furnace',
        recipe=['iron_ore', 'coal'],
        holding=[],
    ),
    dict(
        rule_name='craft_gold_ingot',
        create='gold_ingot',
        action='craft',
        location='furnace',
        recipe=['gold_ore', 'coal'],
        holding=[],
    ),
    dict(
        rule_name='craft_bed',
        create='bed',
        action='craft',
        location='bed_station',
        recipe=['wood_plank', 'wool'],
        holding=[],
    ),
    dict(
        rule_name='craft_boat',
        create='boat',
        action='craft',
        location='boat_station',
        recipe=['wood_plank'],
        holding=[],
    ),
    dict(
        rule_name='craft_bowl1',
        create='bowl',
        action='craft',
        location='food_station',
        recipe=['wood_plank'],
        holding=[],
    ),
    dict(
        rule_name='craft_bowl2',
        create='bowl',
        action='craft',
        location='food_station',
        recipe=['iron_ingot'],
        holding=[],
    ),
    dict(
        rule_name='craft_cooked_potato',
        create='cooked_potato',
        action='craft',
        location='furnace',
        recipe=['potato', 'coal'],
        holding=[],
    ),
    dict(
        rule_name='craft_beetroot_soup',
        create='beetroot_soup',
        action='craft',
        location='food_station',
        recipe=['beetroot', 'bowl'],
        holding=[],
    ),
    dict(
        rule_name='craft_paper',
        create='paper',
        action='craft',
        location='work_station',
        recipe=['sugar_cane'],
        holding=[],
    )
]


@lru_cache()
def get_all_mining_tools() -> List[str]:
    """Return a list of tools that can be used in mining actions."""

    tools = list()
    for rule in MINING_RULES:
        for holding in rule['holding']:
            if holding not in tools:
                tools.append(holding)
    return tools


@lru_cache()
def get_all_mining_locations() -> List[str]:
    """Return a list of locations that can be mined."""

    locations = list()
    for rule in MINING_RULES:
        if rule['location'] not in locations:
            locations.append(rule['location'])
    return locations


@lru_cache()
def get_all_mining_outcomes() -> List[str]:
    """Return a list of outcomes that can be mined."""

    outcomes = list()
    for rule in MINING_RULES:
        if rule['create'] not in outcomes:
            outcomes.append(rule['create'])
    return outcomes


@lru_cache()
def get_all_crafting_ingradients() -> List[str]:
    """Return a list of tools that can be used in crafting actions."""

    tools = list()
    for rule in CRAFTING_RULES:
        for holding in rule['recipe']:
            if holding not in tools:
                tools.append(holding)
    return tools

@lru_cache()
def get_all_crafting_locations(use_only_workstation: bool = False) -> List[str]:
    """Return a list of locations that can be crafted."""

    if use_only_workstation:
        return ['work_station']

    locations = list()
    for rule in CRAFTING_RULES:
        if rule['location'] not in locations:
            locations.append(rule['location'])
    return locations


@lru_cache()
def get_all_crafting_outcomes() -> List[str]:
    """Return a list of outcomes that can be crafted."""

    outcomes = list()
    for rule in CRAFTING_RULES:
        if rule['create'] not in outcomes:
            outcomes.append(rule['create'])
    return outcomes


@lru_cache()
def get_all_locations() -> List[str]:
    """Return a list of all locations."""
    return get_all_mining_locations() + get_all_crafting_locations()