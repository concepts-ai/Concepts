#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : quickaccess.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 05/11/2018
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Optional
from jaclearn.rl.env import RLEnvBase, ProxyRLEnvBase
from jaclearn.rl.space import DiscreteActionSpace
from jaclearn.rl.proxy import LimitLengthProxy

from concepts.benchmark.algorithm_env.sort_envs import ListSortingEnv
from concepts.benchmark.algorithm_env.graph_env import PathGraphEnv

__all__ = ['get_sort_env', 'get_path_env', 'make']


class _MapActionProxy(ProxyRLEnvBase):
    def __init__(self, other, mapping):
        super().__init__(other)
        self._mapping = mapping

    def map_action(self, action):
        assert action < len(self._mapping)
        return self._mapping[action]

    def _get_action_space(self):
        return DiscreteActionSpace(len(self._mapping))

    def _action(self, action):
        return self.proxy.action(self.map_action(action))


def _map_graph_action(p, n, exclude_self=True):
    mapping = [(i, j) for i in range(n) for j in range(n) if (i != j or not exclude_self)]
    p = _MapActionProxy(p, mapping)
    return p


def get_sort_env(n: int, exclude_self: bool = True) -> RLEnvBase:
    """Get a sorting environment with n elements.

    Args:
        n: number of elements.
        exclude_self: whether to exclude swap(i, i) actions.

    Returns:
        the sorting environment.
    """

    env_cls = ListSortingEnv
    p = env_cls(n)
    p = LimitLengthProxy(p, n * 2)
    p = _map_graph_action(p, n, exclude_self=exclude_self)
    return p


def get_path_env(n, dist_range, prob_edge=0.5, directed=False, gen_method='edge', max_episode_len: Optional[int] = None) -> RLEnvBase:
    """Get a path-finding environment with n nodes.

    Args:
        n: number of nodes.
        dist_range: the range of distance between the start and the end.
        prob_edge: the probability of an edge between two nodes.
        directed: whether the graph is directed.
        gen_method: the method to generate the graph. It can be 'edge' or 'dnc' or 'list'.
        max_episode_len: the maximum length of the episode.

    Returns:
        the path-finding environment.
    """

    env_cls = PathGraphEnv
    p = env_cls(n, dist_range, prob_edge, directed=directed, gen_method=gen_method)
    if max_episode_len is not None:
        p = LimitLengthProxy(p, max_episode_len)
    return p


def make(task: str, *args, **kwargs) -> RLEnvBase:
    """Make an environment.

    Args:
        task: the task name. It can be 'sort' or 'path'.
        args: the arguments for the environment.
        kwargs: the keyword arguments for the environment.

    Returns:
        env: the environment.
    """

    if task == 'sort':
        return get_sort_env(*args, **kwargs)
    elif task == 'path':
        return get_path_env(*args, **kwargs)
    else:
        raise ValueError('Unknown task: {}.'.format(task))
