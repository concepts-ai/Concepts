#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : graph.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 05/07/2018
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional

import copy
import numpy as np

import jacinle.random as random

__all__ = ['Graph', 'random_generate_graph', 'random_generate_graph_dnc', 'random_generate_special_graph']


class Graph(object):
    """Store a graph using adjacency matrix."""

    def __init__(self, nr_nodes: int, edges: np.ndarray, coordinates: Optional[np.ndarray] = None):
        """Initialize a graph.

        Args:
            nr_nodes: The Number of nodes in the graph.
            edges: The adjacency matrix of the graph.
        """
        edges = edges.astype('int32')
        assert edges.min() >= 0 and edges.max() <= 1
        self._nr_nodes = nr_nodes
        self._edges = edges
        self._coordinates = coordinates
        self._shortest = None
        self.extra_info = {}

    def get_edges(self):
        """Get the adjacency matrix of the graph. This function will return a copy of the adjacency matrix."""
        return copy.copy(self._edges)

    def get_coordinates(self) -> Optional[np.ndarray]:
        """Get the coordinates of the nodes."""
        return copy.copy(self._coordinates) if self._coordinates is not None else None

    def get_relations(self) -> np.ndarray:
        """Return edges and identity matrix."""
        return np.stack([self.get_edges(), np.eye(self._nr_nodes)], axis=-1)

    def has_edge(self, x, y) -> bool:
        """Return whether there is an edge from node x to node y."""
        return self._edges[x, y] == 1

    def get_out_degree(self) -> int:
        """Return the out degree of each node."""
        return np.sum(self._edges, axis=1)

    def get_shortest(self) -> np.ndarray:
        """Return the length of shortest path between nodes."""
        if self._shortest is not None:
            return self._shortest

        n = self._nr_nodes
        edges = self.get_edges()

        # n + 1 indicates unreachable.
        shortest = np.ones((n, n)) * (n + 1)
        shortest[np.where(edges == 1)] = 1
        # Make sure that shortest[x, x] = 0
        shortest -= shortest * np.eye(n)
        shortest = shortest.astype(np.int32)

        # Floyd Algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if i != j:
                        shortest[i, j] = min(shortest[i, j], shortest[i, k] + shortest[k, j])
        self._shortest = shortest
        return self._shortest

    def get_connectivity(self, k: Optional[int] = None, exclude_self: int = True):
        """Return the k-connectivity for each pair of nodes. It will return the full connectivity matrix if k is None or k < 0.
        When exclude_self is True, the diagonal elements will be 0.

        Args:
            k: the k-connectivity. Default: `None` (full connectivity).
            exclude_self: exclude the diagonal elements. Default: `True`.

        Returns:
            conn: The connectivity matrix.
        """
        shortest = self.get_shortest()
        if k is None or k < 0:
            k = self._nr_nodes
        k = min(k, self._nr_nodes)
        conn = (shortest <= k).astype(np.int32)
        if exclude_self:
            n = self._nr_nodes
            inds = np.where(~np.eye(n, dtype=np.bool_))
            conn = conn[inds]
            conn.resize(n, n - 1)
        return conn


def random_generate_graph(n, p, directed=False) -> Graph:
    """Randomly generate a graph by sampling the existence of each edge.
    Each edge between nodes has the probability `p` (directed) or `p^2` (undirected) to not exist.

    This paradigm is also called the Erdős–Rényi model.

    Args:
        n: The number of nodes in the graph.
        p: the probability that a edge doesn't exist in directed graph.
        directed: Directed or Undirected graph. Default: `False` (undirected).

    Returns:
        graph: Generated graph.
    """
    edges = (random.rand(n, n) < p).astype(np.float32)
    edges -= edges * np.eye(n)
    if not directed:
        edges = np.maximum(edges, edges.T)
    return Graph(n, edges)


def random_generate_graph_dnc(n, p=None, directed=False) -> Graph:
    """Random graph generation method as in DNC, the Differentiable Neural Computer paper.
    Sample :math:`n` nodes in a unit square. sample out-degree (:math:`m`) of each nodes,
    connect to :math:`m` nearest neighbors (Euclidean distance) in the unit square.

    Args:
        n: The number of nodes in the graph.
        p: Control the range of the sample of out-degree. Default: :math:`[1, n // 3]`

            - (float): :math:`[1, int(n * p)]`
            - (int): :math:`[1, p]`
            - (tuple): :math:`[p[0], p[1]]`

        directed: Directed or Undirected graph. Default: `False` (undirected)

    Returns:
        graph: A randomly generated graph.
    """
    edges = np.zeros((n, n), dtype=np.float32)
    pos = random.rand(n, 2)

    def dist(x, y):
        return ((x - y) ** 2).mean()

    if type(p) is tuple:
        lower, upper = p
    else:
        lower = 1
        if p is None:
            upper = n // 3
        elif type(p) is int:
            upper = p
        elif type(p) is float:
            upper = int(n * p)
        else:
            assert False
        upper = max(upper, 1)
    lower = max(lower, 1)
    upper = min(upper, n - 1)

    for i in range(n):
        d = []
        k = random.randint(upper - lower + 1) + lower
        for j in range(n):
            if i != j:
                d.append((dist(pos[i], pos[j]), j))
        d.sort()
        for j in range(k):
            edges[i, d[j][1]] = 1
    if not directed:
        edges = np.maximum(edges, edges.T)
    return Graph(n, edges, pos)


def random_generate_special_graph(n: int, graph_type: str, directed: bool = False) -> Graph:
    """Randomly generate a special type graph.

    For list graph, the nodes are randomly permuted and connected in order. If the graph is directed, the edges are
    directed from the first node to the last node.

    Args:
        n: The number of nodes in the graph.
        graph_type: The type of the graph, e.g. list, tree. Currently only support list.
        directed: Directed or Undirected graph. Default: `False` (undirected).

    Returns:
        graph: Generated graph.
    """
    if graph_type == 'list':
        nodes = random.permutation(n)
        edges = np.zeros((n, n))
        for i in range(n - 1):
            x, y = nodes[i], nodes[i + 1]
            if directed:
                edges[x, y] = 1
            else:
                edges[x, y] = edges[y, x] = 1
        graph = Graph(n, edges)
        graph.extra_info['nodes_list'] = nodes
        return graph
    else:
        assert False, "not supported graph type: {}".format(graph_type)
