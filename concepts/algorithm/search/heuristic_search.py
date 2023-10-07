#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : heuristic_search.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/17/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Generic implementation of A* search."""

from dataclasses import dataclass
from typing import TypeVar, List, Callable, Iterator, Tuple, Set
import heapq as hq

__all__ = ['SearchNode', 'QueueNode', 'run_heuristic_search', 'backtrace_plan']

State = TypeVar('State')
Action = TypeVar('Action')


@dataclass
class SearchNode(object):
    """A node object corresponding to the current search state, containing the state, the parent node, the last action, the cost, and the depth."""

    state: State
    """The current state."""

    parent: 'SearchNode'
    """The parent node."""

    action: Action
    """The action that leads to the current state."""

    cost: float
    """The cost of the path from the root to the current state."""

    g: float
    """The estimated cost-to-go."""


@dataclass
class QueueNode(object):
    """A node object in the queue, containing the priority and the search node."""

    priority: float
    """The priority of the node."""

    node: SearchNode
    """The search node."""

    @property
    def state(self):
        return self.node.state

    def __iter__(self):
        yield self.priority
        yield self.node

    def __lt__(self, other):  # so we don't need the tie-breaker.
        return self.priority < other.priority


def run_heuristic_search(
    initial_state: State,
    check_goal: Callable[[State], bool],
    get_priority: Callable[[State, int], float],
    get_successors: Callable[[State], Iterator[Tuple[Action, State, float]]],
    check_visited: bool = True,
    max_expansions: int = 10000,
    max_depth: int = 1000
):
    """A generic implementation for heuristic search.

    Args:
        initial_state: the initial state.
        check_goal: a function mapping from state to bool, returning True if the state has satisfied the goal.
        get_priority: a function mapping from state to float, returning the priority of the state.
            Smaller priority means higher priority (will be visited first).
        get_successors: a function mapping from state to an iterator of (action, state, cost) tuples.
        check_visited: whether to check if the state has been visited. Set to false if the State representation
            is not hashable.
        max_expansions: the maximum number of expansions.
        max_depth: the maximum depth of the search tree.

    Returns:
        A tuple of (state_sequence, action_sequence, cost_sequence, nr_expansions).

    Raises:
        ValueError: if the search fails.
    """
    queue: List[QueueNode] = list()
    visited: Set[State] = set()

    def push_node(node: SearchNode):
        hq.heappush(queue, QueueNode(get_priority(node.state, node.g), node))
        if check_visited:
            visited.add(node.state)

    root_node = SearchNode(state=initial_state, parent=None, action=None, cost=None, g=0)
    push_node(root_node)
    nr_expansions = 0

    while len(queue) > 0 and nr_expansions < max_expansions:
        priority, node = hq.heappop(queue)

        if node.g > max_depth:
            raise RuntimeError('Failed to find a plan (maximum depth reached).')

        if check_goal(node.state):
            return backtrace_plan(node, nr_expansions)
        nr_expansions += 1
        for action, child_state, cost in get_successors(node.state):
            if check_visited and child_state in visited:
                continue
            child_node = SearchNode(state=child_state, parent=node, action=action, cost=cost, g=node.g + cost)
            push_node(child_node)

    raise RuntimeError('Failed to find a plan (maximum expansion reached).')


def backtrace_plan(node: SearchNode, nr_expansions: int) -> Tuple[List[State], List[Action], List[float], int]:
    """Backtrace the plan from the goal node.

    Args:
        node: a search node where the goal is satisfied.
        nr_expansions: the number of expansions. This value will be returned by this function.

    Returns:
        a tuple of (state_sequence, action_sequence, cost_sequence, nr_expansions).
    """
    state_sequence = []
    action_sequence = []
    cost_sequence = []

    while node.parent is not None:
        action_sequence.insert(0, node.action)
        state_sequence.insert(0, node.state)
        cost_sequence.insert(0, node.cost)
        node = node.parent
    state_sequence.insert(0, node.state)

    return state_sequence, action_sequence, cost_sequence, nr_expansions

