#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : sort_envs.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 06/09/2018
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional

import gym
import numpy as np

from jacinle.utils.tqdm import tqdm
from concepts.benchmark.common.random_env import RandomizedEnv

__all__ = ['ListSortingEnv']


class ListSortingEnv(RandomizedEnv):
    """Env for sorting a random permutation."""

    def __init__(self, nr_numbers: int, np_random: Optional[np.random.RandomState] = None, seed: Optional[int] = None):
        """Initialize the environment.

        Args:
            nr_numbers: The number of numbers in the array.
        """
        super().__init__(np_random=np_random, seed=seed)
        self._nr_numbers = nr_numbers
        self._optimal_nr_steps = None
        self._array = None

        self.action_space = gym.spaces.MultiDiscrete([nr_numbers, nr_numbers])

    @property
    def array(self):
        """The underlying array to be sorted."""
        return self._array

    @property
    def nr_numbers(self):
        """The number of numbers in the array."""
        return self._nr_numbers

    @classmethod
    def make(cls, nr_numbers: int, limit_episode_steps: bool = True, seed: Optional[int] = None) -> gym.Env:
        env = cls(nr_numbers, seed=seed)
        if limit_episode_steps:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=nr_numbers * 2)
        return env

    def reset_nr_numbers(self, n):
        self._nr_numbers = n
        self.action_space = gym.spaces.MultiDiscrete([n, n])

    def reset(self, **kwargs):
        """ Restart: Generate a random permutation. """
        self._array = self.np_random.permutation(self._nr_numbers)
        self._optimal_nr_steps = self._calculate_optimal()
        return self.get_state()

    def step(self, action):
        """Action: Swap the numbers at the index :math:`i` and :math:`j`."""
        a = self._array
        i, j = action
        x, y = a[i], a[j]
        a[i], a[j] = y, x
        for i in range(self._nr_numbers):
            if a[i] != i:
                return self.get_state(), 0, False, {}
        return self.get_state(), 1, True, {}

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

    def oracle_policy(self, state):
        """Oracle policy: Swap the first two numbers that are not sorted."""
        a = self._array
        for i in range(self._nr_numbers):
            if a[i] != i:
                for j in range(i + 1, self._nr_numbers):
                    if a[j] == i:
                        return i, j
        return None

    def generate_data(self, nr_data_points: int):
        data = list()
        for _ in tqdm(range(nr_data_points)):
            obs = self.reset()
            states, actions = [obs], list()
            while True:
                action = self.oracle_policy(obs)
                if action is None:
                    raise RuntimeError('No action found.')
                obs, _, finished, _ = self.step(action)
                states.append(obs)
                actions.append(action)

                if finished:
                    break
            data.append({'states': states, 'actions': actions, 'optimal_steps': self._optimal_nr_steps, 'actual_steps': len(actions)})
        return data

