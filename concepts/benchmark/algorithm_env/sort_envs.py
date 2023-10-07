#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : sort_envs.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 06/09/2018
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np

import jacinle.random as random
from jacinle.utils.meta import notnone_property
from jaclearn.rl.env import SimpleRLEnvBase


class ListSortingEnv(SimpleRLEnvBase):
    """Env for sorting a random permutation."""

    def __init__(self, nr_numbers: int):
        """Initialize the environment.

        Args:
            nr_numbers: The number of numbers in the array.
        """
        super().__init__()
        self._nr_numbers = nr_numbers
        self._array = None

    @notnone_property
    def array(self):
        """The underlying array to be sorted."""
        return self._array

    def get_state(self):
        """Compute the state given the array."""
        x, y = np.meshgrid(self.array, self.array)
        number_relations = np.stack([x < y, x == y, x > y], axis=-1).astype('float')
        index = np.array(list(range(self._nr_numbers)))
        x, y = np.meshgrid(index, index)
        position_relations = np.stack([x < y, x == y, x > y], axis=-1).astype('float')
        return np.concatenate([number_relations, position_relations], axis=-1)

    def _calculate_optimal(self):
        """Calculate the optimal number of steps for sorting the array."""
        a = self._array
        b = [0 for i in range(len(a))]
        cnt = 0
        for i, x in enumerate(a):
            if b[i] == 0:
                j = x
                b[i] = 1
                while b[j] == 0:
                    b[j] = 1
                    j = a[j]
                assert i == j
                cnt += 1
        return len(a) - cnt

    def _restart(self):
        """Restart: Generate a random permutation."""
        self._array = random.permutation(self._nr_numbers)
        self._set_current_state(self.get_state())
        self.optimal = self._calculate_optimal()

    def _action(self, action):
        """Action: Swap the numbers at the index :math:`i` and :math:`j`. Returns: reward, is_over.

        Args:
            action: A tuple of two numbers, indicating the indices to be swapped.

        Returns:
            reward: 1 if the array is sorted, 0 otherwise.
            is_over: True if the array is sorted, False otherwise.
        """
        a = self._array
        i, j = action
        x, y = a[i], a[j]
        a[i], a[j] = y, x
        self._set_current_state(self.get_state())
        for i in range(self._nr_numbers):
            if a[i] != i:
                return 0, False
        return 1, True
