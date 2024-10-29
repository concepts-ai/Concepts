#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : blocksworld.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/03/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional, List

import numpy as np

__all__ = ['Block', 'BlockWorld', 'random_generate_blocks_world']


class Block(object):
    def __init__(self, index, father=None):
        self.index = index
        self.father = father
        self.children = []

    index: int
    father: Optional['Block']
    children: List['Block']

    @property
    def is_ground(self):
        return self.father is None

    @property
    def placeable(self):
        if self.is_ground:
            return True
        return len(self.children) == 0

    @property
    def moveable(self):
        if self.is_ground:
            return False
        return len(self.children) == 0

    def remove_from_father(self):
        assert self in self.father.children
        self.father.children.remove(self)
        self.father = None

    def add_to(self, other):
        self.father = other
        other.children.append(self)


class BlockStorage(object):
    def __init__(self, blocks, random_order=None):
        super().__init__()
        self._blocks = blocks
        self._random_order = None
        self._inv_random_order = None
        self.set_random_order(random_order)

    def __getitem__(self, item):
        if self._random_order is None:
            return self._blocks[item]
        return self._blocks[self._random_order[item]]

    def __len__(self):
        return len(self._blocks)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def raw(self):
        return self._blocks.copy()

    @property
    def random_order(self):
        return self._random_order

    def set_random_order(self, random_order):
        if random_order is None:
            self._random_order = None
            self._inv_random_order = None
            return

        self._random_order = random_order
        self._inv_random_order = sorted(range(len(random_order)), key=lambda x: random_order[x])

    def index(self, i):
        if self._random_order is None:
            return i
        return self._random_order[i]

    def inv_index(self, i):
        if self._random_order is None:
            return i
        return self._inv_random_order[i]

    def permute(self, array):
        if self._random_order is None:
            return array
        return [array[self._random_order[i]] for i in range(len(self._blocks))]


class BlockWorld(object):
    def __init__(self, blocks, random_order=None):
        super().__init__()
        self.blocks = BlockStorage(blocks, random_order)

    @property
    def size(self):
        return len(self.blocks)

    def move(self, x, y):
        if x != y and self.moveable(x, y):
            self.blocks[x].remove_from_father()
            self.blocks[x].add_to(self.blocks[y])

    def moveable(self, x, y):
        return self.blocks[x].moveable and self.blocks[y].placeable

    def get_world_string(self):
        index_mapping = {b.index: i for i, b in enumerate(self.blocks)}
        raw_blocks = self.blocks.raw

        result = ''

        def dfs(block, indent):
            nonlocal result

            result += '{}Block #{}: (IsGround={}, Moveable={}, Placeable={})\n'.format(
                ' ' * (indent * 2), index_mapping[block.index], block.is_ground, block.moveable, block.placeable
            )
            for c in block.children:
                dfs(c, indent + 1)

        dfs(raw_blocks[0], 0)
        return result

    def get_coordinates(self, absolute=False):
        coordinates = [None for _ in range(self.size)]
        raw_blocks = self.blocks.raw

        def dfs(block: Block):
            if block.is_ground:
                coordinates[block.index] = (0, 0)
                for j, c in enumerate(block.children):
                    x = self.blocks.inv_index(c.index) if absolute else j
                    coordinates[c.index] = (x, 1)
                    dfs(c)
            else:
                coor = coordinates[block.index]
                assert coor is not None
                x, y = coor
                for c in block.children:
                    coordinates[c.index] = (x, y + 1)
                    dfs(c)

        dfs(raw_blocks[0])
        coordinates = self.blocks.permute(coordinates)
        return np.array(coordinates)

    def get_on_relation(self):
        on = np.zeros((self.size, self.size), dtype=np.float32)

        def dfs(block):
            if block.is_ground:
                for c in block.children:
                    on[c.index, block.index] = 1
                    dfs(c)
            else:
                for c in block.children:
                    on[c.index, block.index] = 1
                    dfs(c)
        dfs(self.blocks.raw[0])
        return on

    def get_is_ground(self):
        return np.array([block.is_ground for block in self.blocks])

    def get_moveable(self):
        return np.array([block.moveable for block in self.blocks])

    def get_placeable(self):
        return np.array([block.placeable for block in self.blocks])



# similar to random tree generation, randomly sample a valid father for new nodes
def random_generate_blocks_world(nr_blocks, random_order=False, one_stack=False, np_random: Optional[np.random.RandomState] = None):
    if np_random is None:
        np_random = np.random

    blocks = [Block(0, None)]
    leaves = [blocks[0]]

    for i in range(1, nr_blocks + 1):
        other = leaves[np_random.randint(len(leaves))]
        this = Block(i)
        this.add_to(other)
        if not other.placeable or one_stack:
            leaves.remove(other)
        blocks.append(this)
        leaves.append(this)

    order = None
    if random_order:
        order = np_random.permutation(len(blocks))

    return BlockWorld(blocks, random_order=order)

