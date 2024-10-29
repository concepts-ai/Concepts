#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : graph_env.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 04/27/2018
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import numpy as np
import gym
from typing import Optional, Tuple

from jacinle.utils.tqdm import tqdm
from concepts.benchmark.common.random_env import RandomizedEnv
from concepts.benchmark.algorithm_env.graph import random_generate_graph, random_generate_graph_dnc, random_generate_special_graph

__all__ = ['GraphEnvBase', 'GraphPathEnv']


class GraphEnvBase(RandomizedEnv):
    """Graph Env Base."""

    def __init__(self, nr_nodes: int, p: float = 0.5, directed: bool = False, gen_method: str = 'edge', np_random: Optional[np.random.RandomState] = None, seed: Optional[int] = None):
        """Initialize the environment.

        Args:
            nr_nodes: the number of nodes in the graph.
            p: parameter for random generation. (Default: 0.5)
                - (edge method): The probability that an edge doesn't exist in directed graph.
                - (dnc method): Control the range of the sample of out-degree.
                - other methods: Unused.
            directed: directed or Undirected graph. Default: `False` (undirected)
            gen_method: use which method to randomly generate a graph.
                - 'edge': By sampling the existence of each edge.
                - 'dnc': Sample out-degree (:math:`m`) of each node, and link to nearest neighbors in the unit square.
                - 'list': generate a chain-like graph.
        """
        super().__init__(np_random, seed)

        self._nr_nodes = nr_nodes
        self._p = p
        self._directed = directed
        self._gen_method = gen_method
        self._graph = None

    @property
    def action_space(self):
        raise NotImplementedError

    @property
    def observation_space(self):
        raise NotImplementedError

    @property
    def graph(self):
        """The generated graph."""
        return self._graph

    def _gen_random_graph(self):
        """ generate the graph by specified method. """
        n = self._nr_nodes
        p = self._p
        if self._gen_method == 'edge':
            self._graph = random_generate_graph(n, p, self._directed)
        elif self._gen_method == 'dnc':
            self._graph = random_generate_graph_dnc(n, p, self._directed)
        else:
            self._graph = random_generate_special_graph(n, self._gen_method, self._directed)


class GraphPathEnv(GraphEnvBase):
    """Env for Finding a path from starting node to the destination."""

    def __init__(self, nr_nodes: int, dist_range: Tuple[int, int], p: float = 0.5, directed: bool = False, gen_method: str = 'edge', np_random: Optional[np.random.RandomState] = None, seed: Optional[int] = None):
        """Initialize the environment.

        Args:
            nr_nodes: the number of nodes in the graph.
            dist_range: the sampling range of distance between starting node and the destination.
            p: parameter for random generation. (Default: 0.5)
                - (edge method): The probability that an edge doesn't exist in directed graph.
                - (dnc method): Control the range of the sample of out-degree.
                - other methods: Unused.
            directed: directed or Undirected graph. Default: `False` (undirected)
            gen_method: use which method to randomly generate a graph.
                - 'edge': By sampling the existence of each edge.
                - 'dnc': Sample out-degree (:math:`m`) of each node, and link to the nearest neighbors in the unit square.
                - 'list': generate a chain-like graph.
            np_random: random state. If None, a new random state will be created based on the seed.
            seed: random seed. If None, a randomly chosen seed will be used.
        """
        super().__init__(nr_nodes, p, directed, gen_method, np_random=np_random, seed=seed)
        self._dist_range = dist_range

        self._dist = None
        self._dist_matrix = None
        self._task = None
        self._current = None
        self._steps = None

        self.action_space = gym.spaces.MultiDiscrete([nr_nodes, nr_nodes])

    @classmethod
    def make(cls, n: int, dist_range: Tuple[int, int], p: float = 0.5, directed: bool = False, gen_method: str = 'edge', seed: Optional[int] = None) -> gym.Env:
        env = cls(n, dist_range, p=p, directed=directed, gen_method=gen_method, seed=seed)
        return env

    @property
    def dist(self) -> int:
        return self._dist

    def reset_nr_nodes(self, nr_nodes: int):
        self._nr_nodes = nr_nodes
        self.action_space = gym.spaces.MultiDiscrete([nr_nodes, nr_nodes])

    def reset(self, **kwargs):
        """Restart the environment."""
        self._dist = self._gen_random_distance()
        self._task = None
        while True:
            self._gen_random_graph()
            self._dist_matrix = self._graph.get_shortest()
            self._task = self._gen_random_task()
            if self._task is not None:
                break
        self._current = self._task[0]
        self._steps = 0
        return self.get_state()

    def step(self, action):
        """Move to the target node from the current node if has_edge(current -> target)."""
        if self._current == self._task[1]:
            return self.get_state(), 1, True, {}
        if self._graph.has_edge(self._current, action):
            self._current = action
        if self._current == self._task[1]:
            return self.get_state(), 1, True, {}
        self._steps += 1
        if self._steps >= self.dist:
            return self.get_state(), 0, True, {}
        return self.get_state(), 0, False, {}

    def _gen_random_distance(self):
        lower, upper = self._dist_range
        upper = min(upper, self._nr_nodes - 1)
        return self.np_random.randint(upper - lower + 1) + lower

    def _gen_random_task(self):
        """Sample the starting node and the destination according to the distance."""
        st, ed = np.where(self._dist_matrix == self._dist)
        if len(st) == 0:
            return None
        ind = self.np_random.randint(len(st))
        return st[ind], ed[ind]

    def get_state(self):
        relation = self._graph.get_edges()
        current_state = np.zeros_like(relation)
        current_state[self._current, :] = 1
        return np.stack([relation, current_state], axis=-1)

    def oracle_policy(self, state):
        """Oracle policy: Swap the first two numbers that are not sorted."""
        current = self._current
        target = self._task[1]
        if current == target:
            return 0
        possible_actions = state[current, :, 0] == 1
        possible_actions = possible_actions & self._dist_matrix[:, target] < self._dist_matrix[current, target]
        if np.sum(possible_actions) == 0:
            raise RuntimeError('No action found.')
        return self.np_random.choice(np.where(possible_actions)[0])

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
            data.append({'states': states, 'actions': actions, 'optimal_steps': self._dist, 'actual_steps': len(actions)})
        return data



