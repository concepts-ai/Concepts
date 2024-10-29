#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : language_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/09/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import jacinle
from typing import Any, Tuple, Dict

"""
include/align.h
#define A_NONE (-128) /* the value range of type */
#define A_CHAOTIC (-1)
#define A_NEUTRAL 0
#define A_LAWFUL 1
std::unordered_map<int64_t, std::string> alignment_map{
  {A_NONE, "None"},
  {A_LAWFUL, "Lawful"},
  {A_NEUTRAL, "Neutral"},
  {A_CHAOTIC, "Chaotic"}};
      
include/hack.h
enum hunger_state_types {
    SATIATED   = 0,
    NOT_HUNGRY = 1,
    HUNGRY     = 2,
    WEAK       = 3,
    FAINTING   = 4,
    FAINTED    = 5,
    STARVED    = 6
};
std::unordered_map<int64_t, std::string> hunger_map{
  {SATIATED, "Satiated"}, {NOT_HUNGRY, "Not Hungry"}, {HUNGRY, "Hungry"},
  {WEAK, "Weak"},         {FAINTING, "Fainting"},     {FAINTED, "Fainted"},
  {STARVED, "Starved"}};

enum encumbrance_types {
    UNENCUMBERED = 0,
    SLT_ENCUMBER = 1, /* Burdened */
    MOD_ENCUMBER = 2, /* Stressed */
    HVY_ENCUMBER = 3, /* Strained */
    EXT_ENCUMBER = 4, /* Overtaxed */
    OVERLOADED   = 5  /* Overloaded */
}; 
std::unordered_map<int64_t, std::string> encumbrance_map{
  {UNENCUMBERED, "Unencumbered"}, {SLT_ENCUMBER, "Burdened"},
  {MOD_ENCUMBER, "Stressed"},     {HVY_ENCUMBER, "Strained"},
  {EXT_ENCUMBER, "Overtaxed"},    {OVERLOADED, "Overloaded"}};


// include/botl.h
#define BL_MASK_STONE           0x00000001L
#define BL_MASK_SLIME           0x00000002L
#define BL_MASK_STRNGL          0x00000004L
#define BL_MASK_FOODPOIS        0x00000008L
#define BL_MASK_TERMILL         0x00000010L
#define BL_MASK_BLIND           0x00000020L
#define BL_MASK_DEAF            0x00000040L
#define BL_MASK_STUN            0x00000080L
#define BL_MASK_CONF            0x00000100L
#define BL_MASK_HALLU           0x00000200L
#define BL_MASK_LEV             0x00000400L
#define BL_MASK_FLY             0x00000800L
#define BL_MASK_RIDE            0x00001000L
#define BL_MASK_BITS            13 /* number of mask bits that can be set */
std::unordered_map<int64_t, std::string> condition_map = {
  {BL_MASK_STONE, "Stoned"},
  {BL_MASK_SLIME, "Slimed"},
  {BL_MASK_STRNGL, "Strangled"},
  {BL_MASK_FOODPOIS, "Food Poisoning"},
  {BL_MASK_TERMILL, "Terminally Ill"},
  {BL_MASK_BLIND, "Blind"},
  {BL_MASK_DEAF, "Deaf"},
  {BL_MASK_STUN, "Stunned"},
  {BL_MASK_CONF, "Confused"},
  {BL_MASK_HALLU, "Hallucinating"},
  {BL_MASK_LEV, "Levitating"},
  {BL_MASK_FLY, "Flying"},
  {BL_MASK_RIDE, "Riding"},
};

py::bytes NLELanguageObsv::text_blstats(py::array_t<int64_t> blstats) {
  py::buffer_info blstats_buffer = blstats.request();
  int64_t *blstats_data = reinterpret_cast<int64_t *>(blstats_buffer.ptr);

  std::string alignment_str = alignment_map[blstats_data[26]];
  std::string hunger_str = hunger_map[blstats_data[21]];
  std::string encumbrance_str = encumbrance_map[blstats_data[22]];

  std::vector<std::string> conditions;
  for (const auto &[mask, condition] : condition_map) {
    if (blstats_data[25] & mask) {
      conditions.push_back(condition);
    }
  }

  std::string condition;
  if (conditions.empty()) {
    condition = "None";
  } else if (conditions.size() == 1) {
    condition = conditions[0];
  } else {
    condition = conditions[0];
    for (auto it = ++conditions.begin(); it != conditions.end(); ++it) {
      condition += " " + *it;
    }
  }

  std::stringstream ss;
  ss << "Strength: " << blstats_data[3] << "/" << blstats_data[2] << "\n"
     << "Dexterity: " << blstats_data[4] << "\n"
     << "Constitution: " << blstats_data[5] << "\n"
     << "Intelligence: " << blstats_data[6] << "\n"
     << "Wisdom: " << blstats_data[7] << "\n"
     << "Charisma: " << blstats_data[8] << "\n"
     << "Depth: " << blstats_data[12] << "\n"
     << "Gold: " << blstats_data[13] << "\n"
     << "HP: " << blstats_data[10] << "/" << blstats_data[11] << "\n"
     << "Energy: " << blstats_data[14] << "/" << blstats_data[15] << "\n"
     << "AC: " << blstats_data[16] << "\n"
     << "XP: " << blstats_data[18] << "/" << blstats_data[19] << "\n"
     << "Time: " << blstats_data[20] << "\n"
     << "Position: " << blstats_data[0] << "|" << blstats_data[1] << "\n"
     << "Hunger: " << hunger_str << "\n"
     << "Monster Level: " << blstats_data[17] << "\n"
     << "Encumbrance: " << encumbrance_str << "\n"
     << "Dungeon Number: " << blstats_data[23] << "\n"
     << "Level Number: " << blstats_data[24] << "\n"
     << "Score: " << blstats_data[9] << "\n"
     << "Alignment: " << alignment_str << "\n"
     << "Condition: " << condition;

  return py::bytes(ss.str());
}
"""

g_alignment_map = {
    -128: "None",
    -1: "Chaotic",
    0: "Neutral",
    1: "Lawful"
}

g_hunger_map = {
    0: "Satiated",
    1: "Not Hungry",
    2: "Hungry",
    3: "Weak",
    4: "Fainting",
    5: "Fainted",
    6: "Starved"
}

g_encumbrance_map = {
    0: "Unencumbered",
    1: "Burdened",
    2: "Stressed",
    3: "Strained",
    4: "Overtaxed",
    5: "Overloaded"
}

g_condition_map = {
    1 << 0: "Stoned",
    1 << 1: "Slimed",
    1 << 2: "Strangled",
    1 << 3: "Food Poisoning",
    1 << 4: "Terminally Ill",
    1 << 5: "Blind",
    1 << 6: "Deaf",
    1 << 7: "Stunned",
    1 << 8: "Confused",
    1 << 9: "Hallucinating",
    1 << 10: "Levitating",
    1 << 11: "Flying",
    1 << 12: "Riding"
}

def blstat_to_dict(blstat: Tuple[int, ...]) -> Dict[str, Any]:
    alignment_str = g_alignment_map[blstat[26]]
    hunger_str = g_hunger_map[blstat[21]]
    encumbrance_str = g_encumbrance_map[blstat[22]]

    conditions = [condition for mask, condition in g_condition_map.items() if blstat[25] & mask]
    condition = 'None' if not conditions else ' '.join(conditions)

    return {
        "Strength": f"{blstat[3]}/{blstat[2]}",
        "Strength Value": blstat[3],
        "Strength Max": blstat[2],
        "Dexterity": blstat[4],
        "Constitution": blstat[5],
        "Intelligence": blstat[6],
        "Wisdom": blstat[7],
        "Charisma": blstat[8],
        "Depth": blstat[12],
        "Gold": blstat[13],
        "HP": f"{blstat[10]}/{blstat[11]}",
        "HP Value": blstat[10],
        "HP Max": blstat[11],
        "Energy": f"{blstat[14]}/{blstat[15]}",
        "Energy Value": blstat[14],
        "Energy Max": blstat[15],
        "AC": blstat[16],
        "XP": f"{blstat[18]}/{blstat[19]}",
        "XP Value": blstat[18],
        "XP Max": blstat[19],
        "Time": blstat[20],
        "Position": f"{blstat[0]}|{blstat[1]}",
        "Position X": blstat[0],
        "Position Y": blstat[1],
        "Hunger": hunger_str,
        "Monster Level": blstat[17],
        "Encumbrance": encumbrance_str,
        "Dungeon Number": blstat[23],
        "Level Number": blstat[24],
        "Score": blstat[9],
        "Alignment": alignment_str,
        "Condition": condition
    }


def inventory_to_dict(inv_letters: Tuple[int, ...], inv_strs: Tuple[str, ...]) -> Dict[str, str]:
    inventory_strings = list()
    for identifier, s in zip(inv_letters, inv_strs):
        if identifier == 0:
            continue
        s = ''.join(chr(x) for x in s if x > 0).strip()
        if len(s) == 0:
            continue
        inventory_strings.append((chr(identifier), s))

    return dict(inventory_strings)


def char_array_to_str(char_array: Tuple[int, ...]) -> str:
    return ''.join(chr(x) for x in char_array if x > 0).strip()


def pretty_format_additional_observations(observation: Dict[str, Any]):
    if 'blstats' in observation:
        print("Player Status:")
        table = blstat_to_dict(observation['blstats'])
        print(jacinle.tabulate(table.items(), headers=['Attribute', 'Value']))
    if 'inv_letters' in observation:
        print("Inventory:")
        table = inventory_to_dict(observation['inv_letters'], observation['inv_strs'])
        print(jacinle.tabulate(table.items(), headers=['ID', 'Name']))
    # if 'message' in observation:
    #     print("Message:")
    #     print(char_array_to_str(observation['message']))
    print('-' * 120)
