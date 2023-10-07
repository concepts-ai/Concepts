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

import jacinle.random as random
from jacinle.utils.meta import notnone_property
from jaclearn.rl.env import SimpleRLEnvBase

from concepts.benchmark.algorithm_env.graph import random_generate_graph, random_generate_graph_dnc, random_generate_special_graph

__all__ = ['GraphEnvBase', 'PathGraphEnv']


class GraphEnvBase(SimpleRLEnvBase):
    """Graph Env Base."""

    def __init__(self, nr_nodes, p=0.5, directed=False, gen_method='edge'):
        """Initialize the environment.

        Args:
            n: The number of nodes in the graph.
            p: Parameter for random generation. (Default 0.5)
                (edge method): The probability that a edge doesn't exist in directed graph.
                (dnc method): Control the range of the sample of out-degree.
                other methods: Unused.
            directed: Directed or Undirected graph. Default: `False`(undirected)
            gen_method: Use which method to randomly generate a graph.
                'edge': By sampling the existance of each edge.
                'dnc': Sample out-degree (:math:`m`) of each nodes, and link to nearest neighbors in the unit square.
                'list': generate a chain-like graph.
        """

        super().__init__()
        self._nr_nodes = nr_nodes
        self._p = p
        self._directed = directed
        self._gen_method = gen_method
        self._graph = None

    @notnone_property
    def graph(self):
        """The generated graph."""
        return self._graph

    def _restart(self):
        """ Restart the environment. """
        self._gen_graph()

    def _gen_graph(self):
        """ generate the graph by specified method. """
        n = self._nr_nodes
        p = self._p
        if self._gen_method == 'edge':
            self.graph = random_generate_graph(n, p, self._directed)
        elif self._gen_method == 'dnc':
            self.graph = random_generate_graph_dnc(n, p, self._directed)
        else:
            self._graph = random_generate_special_graph(n, self._gen_method, self._directed)


class PathGraphEnv(GraphEnvBase):
    """Env for Finding a path from starting node to the destination."""
    def __init__(self, nr_nodes, dist_range, p=0.5, directed=False, gen_method='edge'):
        """Initialize the environment.

        Args:
            nr_nodes: The number of nodes in the graph.
            p: Parameter for random generation. (Default: 0.5)
                (edge method): The probability that an edge doesn't exist in directed graph.
                (dnc method): Control the range of the sample of out-degree.
                other methods: Unused.
            directed: Directed or Undirected graph. Default: `False` (undirected)
            gen_method: Use which method to randomly generate a graph.
                'edge': By sampling the existance of each edge.
                'dnc': Sample out-degree (:math:`m`) of each nodes, and link to nearest neighbors in the unit square.
                'list': generate a chain-like graph.
            dist_range: The sampling range of distance between starting node and the destination.
        """
        super().__init__(nr_nodes, p, directed, gen_method)
        self._dist_range = dist_range

    @property
    def dist(self):
        """The distance between starting node and the destination."""
        return self._dist

    def _restart(self):
        """ Restart the environment. """
        super()._restart()
        self._dist = self._sample_dist()
        self._task = None
        while True:
            self._task = self._gen()
            if self._task is not None:
                break
            # If fail to find two nodes with sampled distance, generate another graph.
            self._gen_graph()
        self._current = self._task[0]
        self._set_current_state(self._task)
        self._steps = 0

    def _sample_dist(self):
        lower, upper = self._dist_range
        upper = min(upper, self._nr_nodes - 1)
        return random.randint(upper - lower + 1) + lower

    def _gen(self):
        """ Sample the starting node and the destination according to the distance. """
        dist_matrix = self._graph.get_shortest()
        st, ed = np.where(dist_matrix == self.dist)
        if len(st) == 0:
            return None
        ind = random.randint(len(st))
        return st[ind], ed[ind]

    def _action(self, target):
        """
            Move to the target node from current node if has_edge(current -> target).
            Returns: reward, is_over
        """
        if self._current == self._task[1]:
            return 1, True
        if self._graph.has_edge(self._current, target):
            self._current = target
        self._set_current_state((self._current, self._task[1]))
        if self._current == self._task[1]:
            return 1, True
        self._steps += 1
        if self._steps >= self.dist:
            return 0, True
        return 0, False

