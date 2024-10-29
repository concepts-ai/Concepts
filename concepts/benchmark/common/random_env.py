#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : random_env.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/02/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Any, Optional, Tuple

import numpy as np
from gym.core import Env


class RandomizedEnv(Env):
    def __init__(self, np_random: Optional[np.random.RandomState] = None, seed: Optional[int] = None):
        if np_random is None:
            if seed is None:
                self._np_random = np.random.RandomState()
            else:
                self._np_random = np.random.RandomState(seed)
        else:
            self._np_random = np_random

    _np_random: np.random.RandomState

    @property
    def np_random(self) -> np.random.RandomState:
        return self._np_random

    def seed(self, seed=None):
        self._np_random.seed(seed)

    @property
    def action_space(self):
        raise NotImplementedError

    @property
    def observation_space(self):
        raise NotImplementedError

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        """Run one timestep of the environment's dynamics.
        When end of episode is reached, you are responsible for calling `reset()` to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action: an action provided by the environment

        Returns:
            observation: agent's observation of the current environment
            reward: amount of reward returned after previous action
            done: whether the episode has ended, in which case further step() calls will return undefined results
            info: contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

